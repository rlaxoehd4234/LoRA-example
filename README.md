# Qwen2.5 LoRA 8GB 데모 (mock data)

`rlaxoehd4234/sllm` 실습을 8GB RAM 안에서 돌리기 위해 만든 독립 실행 가능한
축소판입니다. 실제 회사 규정 대신 **가상의 HR 규정(mock data)**을 씁니다 —
파이프라인 자체를 검증하는 게 목적입니다.

## 구성

```
.
├── requirements.txt
├── mock_data/
│   ├── hr_sft_train.jsonl        # 16건 — 가상 HR 규정 Q&A
│   ├── hr_sft_validation.jsonl   # 5건
│   └── hr_eval.jsonl             # 6건 (known_policy 4 + unknown_policy 2)
├── scripts/
│   ├── train_lora.py             # rank=8 LoRA SFT, 8GB 맞춤 설정
│   └── evaluate_model.py         # Base vs SFT, Keyword/F1/Hallucination 채점
├── models/hr-qwen-lora/          # 학습 후 adapter가 저장되는 곳 (최초엔 비어있음)
└── outputs/                       # 학습/평가 결과 저장 (최초엔 비어있음)
```

## 실행

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 1. 학습

```bash
SFT_MAX_LENGTH=384 \
SFT_EPOCHS=2 \
SFT_LEARNING_RATE=1e-4 \
SFT_GRAD_ACCUM=4 \
python scripts/train_lora.py
```

CUDA/MPS/CPU를 자동 감지합니다. 끝나면 `models/hr-qwen-lora/`에 LoRA
adapter가, `outputs/training_summary.json`에 최종 loss 등 요약이 저장됩니다.

### 2. 평가

```bash
python scripts/evaluate_model.py
```

`outputs/evaluation_results.jsonl`(문항별 상세)과
`outputs/evaluation_summary.json`(SFT가 Base를 이긴 횟수 등 요약)이 생성됩니다.

## mock_data가 진짜 데이터와 다른 점

- 회사명(Human AI Corporation)과 인사 규정 문구는 전부 이 데모를 위해
  새로 지어낸 것입니다 — 실제 sllm 저장소의 규정 내용과 다릅니다.
- 개수도 줄였습니다 (582→16건) — 파이프라인이 도는지 빠르게 확인하는 용도라,
  실제 성능을 보려면 원본 저장소의 진짜 데이터셋으로 교체하세요. 형식
  (`{"messages": [...]}`, eval의 `category`/`must_include`/`must_not_include`/
  `reference_answer`)은 원본과 동일해서 파일만 갈아끼우면 됩니다.

## 8GB 설정이 어디에 반영되어 있는지

| 가이드에서 정한 것 | 코드 위치 |
|---|---|
| rank=8, alpha=16, target_modules=[q_proj, v_proj] | `train_lora.py`의 `LoraConfig` |
| batch=1 + grad_accum=4 | `SFTConfig`의 `per_device_train_batch_size`/`gradient_accumulation_steps` |
| gradient_checkpointing 필수 | `SFTConfig(gradient_checkpointing=True, ...)` |
| max_length=384 | `SFTConfig`의 `max_length` |
| CUDA만 bf16, MPS/CPU는 fp32 | `train_lora.py`/`evaluate_model.py`의 `detect_device_type()` 분기 |
