"""
Compare the Base model against the LoRA-tuned (SFT) model on mock_data/hr_eval.jsonl.

Base vs SFT without loading two models: the LoRA adapter is attached once,
then temporarily disabled (`model.disable_adapter()`) to get the "Base"
answer, and re-enabled for the "SFT" answer — same weights in memory the
whole time, which matters at an 8GB budget.

Three scores per answer:
  - keyword_score   — fraction of `must_include` terms actually present
                       (a `must_not_include` hit that shouldn't be there costs -0.5)
  - token_f1_score  — token-overlap F1 against `reference_answer`
  - hallucination_score — only scored when category == "unknown_policy":
                       1.0 if the answer only uses safe hedging language,
                       0.5 if it mixes safe + risky claims, 0.0 if it
                       confidently states something not in policy

Weighting (calculate_total_score):
  - known_policy_paraphrase : keyword 70% + F1 30%
  - unknown_policy          : keyword 30% + F1 20% + hallucination 50%
                              (safety dominates when the model doesn't know)

Usage:
    python scripts/evaluate_model.py
"""
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = ROOT / "models" / "hr-qwen-lora"
EVAL_FILE = ROOT / "mock_data" / "hr_eval.jsonl"
OUTPUTS_DIR = ROOT / "outputs"

SAFE_HEDGE_PATTERNS = [
    r"확인이? 필요", r"확인되지 않", r"인사팀.*문의", r"알 수 없", r"정확한 (안내|규정)",
]
RISKY_CLAIM_PATTERNS = [
    r"가능합니다", r"허용됩니다", r"문제없습니다", r"전액 지원", r"자유롭게",
]


def validate_adapter():
    cfg = ADAPTER_PATH / "adapter_config.json"
    if not cfg.exists():
        raise FileNotFoundError(
            f"No adapter found at {ADAPTER_PATH} — run scripts/train_lora.py first."
        )


def detect_device_type() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_models(device_type: str):
    dtype = torch.bfloat16 if device_type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, device_map={"": device_type} if device_type != "cpu" else None
    )
    tuned = PeftModel.from_pretrained(base, str(ADAPTER_PATH))
    tuned.eval()
    return tuned, tokenizer, device_type


def generate_answer(model, tokenizer, messages: list[dict], device_type: str, max_new_tokens=200) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    if device_type != "cpu":
        inputs = {k: v.to(device_type) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # deterministic, so Base/SFT are compared fairly
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def keyword_score(answer: str, must_include: list[str], must_not_include: list[str]) -> float:
    if not must_include and not must_not_include:
        return 1.0
    hits = sum(1 for kw in must_include if kw in answer)
    score = hits / len(must_include) if must_include else 1.0
    penalty = 0.5 * sum(1 for kw in must_not_include if kw in answer)
    return max(0.0, score - penalty)


def _normalize(text: str) -> list[str]:
    text = re.sub(r"[^\w가-힣\s]", " ", text)
    return [t for t in text.split() if t]


def token_f1_score(answer: str, reference: str) -> float:
    ans_tokens, ref_tokens = set(_normalize(answer)), set(_normalize(reference))
    if not ans_tokens or not ref_tokens:
        return 0.0
    overlap = len(ans_tokens & ref_tokens)
    precision = overlap / len(ans_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def hallucination_score(answer: str, category: str) -> float | None:
    if category != "unknown_policy":
        return None
    has_safe = any(re.search(p, answer) for p in SAFE_HEDGE_PATTERNS)
    has_risky = any(re.search(p, answer) for p in RISKY_CLAIM_PATTERNS)
    if has_safe and not has_risky:
        return 1.0
    if has_safe and has_risky:
        return 0.5
    if has_risky:
        return 0.0
    return 0.5  # neither pattern matched — ambiguous, treat as middling


def calculate_total_score(category: str, kw: float, f1: float, hallu: float | None) -> float:
    if category == "unknown_policy":
        return 0.3 * kw + 0.2 * f1 + 0.5 * (hallu if hallu is not None else 0.5)
    return 0.7 * kw + 0.3 * f1


def determine_winner(base_total: float, sft_total: float) -> str:
    if abs(base_total - sft_total) <= 0.01:
        return "tie"
    return "sft" if sft_total > base_total else "base"


def main():
    validate_adapter()
    device_type = detect_device_type()
    model, tokenizer, device_type = load_models(device_type)
    print(f"[evaluate_model] device={device_type} adapter={ADAPTER_PATH}")

    items = [json.loads(l) for l in EVAL_FILE.read_text().splitlines() if l.strip()]
    results = []

    for item in items:
        with model.disable_adapter():
            base_answer = generate_answer(model, tokenizer, item["messages"], device_type)
        sft_answer = generate_answer(model, tokenizer, item["messages"], device_type)

        row = {"question": item["messages"][-1]["content"], "category": item["category"]}
        for label, answer in [("base", base_answer), ("sft", sft_answer)]:
            kw = keyword_score(answer, item.get("must_include", []), item.get("must_not_include", []))
            f1 = token_f1_score(answer, item["reference_answer"])
            hallu = hallucination_score(answer, item["category"])
            total = calculate_total_score(item["category"], kw, f1, hallu)
            row[f"{label}_answer"] = answer
            row[f"{label}_keyword"] = round(kw, 3)
            row[f"{label}_f1"] = round(f1, 3)
            row[f"{label}_hallucination"] = hallu
            row[f"{label}_total"] = round(total, 3)

        row["winner"] = determine_winner(row["base_total"], row["sft_total"])
        results.append(row)
        print(f"  [{item['category']}] {row['question'][:30]}... -> base={row['base_total']} sft={row['sft_total']} ({row['winner']})")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_DIR / "evaluation_results.jsonl", "w") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "n_items": len(results),
        "avg_base_total": round(sum(r["base_total"] for r in results) / len(results), 3),
        "avg_sft_total": round(sum(r["sft_total"] for r in results) / len(results), 3),
        "sft_wins": sum(1 for r in results if r["winner"] == "sft"),
        "base_wins": sum(1 for r in results if r["winner"] == "base"),
        "ties": sum(1 for r in results if r["winner"] == "tie"),
    }
    (OUTPUTS_DIR / "evaluation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[evaluate_model] summary -> {summary}")


if __name__ == "__main__":
    main()
