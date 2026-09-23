"""
Pure scoring functions for evaluate_model.py — split out into their own
module (no torch/transformers imports) so tests/test_scoring.py can run
fast, without needing the model dependencies installed at all.

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
"""
import re

SAFE_HEDGE_PATTERNS = [
    r"확인이? 필요", r"확인되지 않", r"인사팀.*문의", r"알 수 없", r"정확한 (안내|규정)",
]
RISKY_CLAIM_PATTERNS = [
    r"가능합니다", r"허용됩니다", r"문제없습니다", r"전액 지원", r"자유롭게",
]


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
