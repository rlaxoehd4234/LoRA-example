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
│   ├── hr_sft_validation.jsonl   # 5건 — 학습 중 loss 확인용 (골드셋 아님)
│   └── hr_eval.jsonl             # 10건 — 골드셋 (known_policy 7 + unknown_policy 3)
├── scripts/
│   ├── train_lora.py             # rank=8 LoRA SFT, 8GB 맞춤 설정
│   ├── evaluate_model.py         # Base vs SFT 비교 실행 (모델 로드 필요)
│   └── scoring.py                # 채점 로직만 분리 (Keyword/F1/Hallucination) — 모델 의존성 없음
├── tests/
│   └── test_scoring.py           # scoring.py 유닛 테스트 (21개) — 모델 없이 몇 초 안에 실행
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

### 3. 테스트 (채점 로직 검증 — 모델 필요 없음)

```bash
pytest tests/ -v
```

`evaluate_model.py`가 매기는 점수가 정확한지 먼저 이걸로 확인하세요 — 채점
로직이 틀렸으면 "SFT가 5/6 이겼다" 같은 결과 자체가 의미 없어집니다. 모델을
전혀 안 띄우고 `scripts/scoring.py`의 순수 함수만 테스트하기 때문에 pytorch
없이도 몇 초 안에 끝납니다.

| 테스트 대상 | 확인하는 것 |
|---|---|
| `keyword_score` | 필수 단어 다 있으면 1.0, 없으면 0.0, 금지 단어 있으면 감점(0 밑으로는 안 내려감) |
| `token_f1_score` | 완전히 같으면 1.0, 완전히 다르면 0.0, 일부 겹치면 그 사이 |
| `hallucination_score` | `unknown_policy`가 아니면 `None`, 안전 표현만 있으면 1.0, 위험 표현 섞이면 0.5, 위험 표현만 있으면 0.0 |
| `calculate_total_score` | `unknown_policy`에서 안전성(50%)이 키워드·F1(30%+20%)보다 실제로 더 큰 영향을 주는지 |
| `determine_winner` | 점수 차가 0.01 이내면 무승부 판정하는지 |

## 테스트는 이 프로젝트에서 두 층위로 나뉩니다

1. **골드셋 (`mock_data/hr_eval.jsonl`, 10건)** — "모델의 답변이 맞는가"를 검증. 사람이 정답(`reference_answer`)과 채점 기준(`must_include`/`must_not_include`)을 미리 정해둔 것.
2. **유닛 테스트 (`tests/test_scoring.py`, 21건)** — "채점 로직 자체가 맞는가"를 검증. 골드셋으로 아무리 평가해도 채점 함수에 버그가 있으면 결과를 믿을 수 없어서, 이 둘은 서로 다른 걸 보장합니다.

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
