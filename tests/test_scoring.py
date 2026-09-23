"""
Unit tests for scripts/scoring.py — these test the grading logic itself,
independent of the model. Run before trusting any evaluation_summary.json
number: if the scorer is wrong, a "SFT wins 5/6" result means nothing.

Run:
    pip install pytest
    pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scoring import calculate_total_score, determine_winner, hallucination_score, keyword_score, token_f1_score  # noqa: E402


# ---- keyword_score ----

def test_keyword_score_all_present():
    assert keyword_score("연차는 15일 부여됩니다", ["15일"], []) == 1.0


def test_keyword_score_missing():
    assert keyword_score("연차는 넉넉히 드립니다", ["15일"], []) == 0.0


def test_keyword_score_partial():
    score = keyword_score("주 2회 재택근무가 가능하며 승인이 필요합니다", ["주 2회", "승인"], [])
    assert score == 1.0  # both present


def test_keyword_score_forbidden_term_penalizes():
    # must_include satisfied, but a must_not_include term also appears
    score = keyword_score("100만원 지원되고 200만원만 되는 건 아닙니다", ["100만원"], ["200만원만"])
    assert score == 0.5  # 1.0 - 0.5 penalty


def test_keyword_score_no_constraints_defaults_to_full_credit():
    assert keyword_score("아무 말이나", [], []) == 1.0


def test_keyword_score_never_goes_negative():
    # two forbidden terms both present, zero must_include hits
    score = keyword_score("가능합니다 허용됩니다", ["필수단어"], ["가능합니다", "허용됩니다"])
    assert score == 0.0  # clamped, not -1.0


# ---- token_f1_score ----

def test_f1_identical_strings_is_one():
    assert token_f1_score("연차는 15일입니다", "연차는 15일입니다") == 1.0


def test_f1_no_overlap_is_zero():
    assert token_f1_score("완전히 다른 문장", "전혀 상관없는 답") == 0.0


def test_f1_partial_overlap_between_zero_and_one():
    score = token_f1_score("연차는 15일 부여됩니다", "연차는 15일이며 최대 25일까지 늘어납니다")
    assert 0.0 < score < 1.0


def test_f1_empty_answer_is_zero():
    assert token_f1_score("", "연차는 15일입니다") == 0.0


# ---- hallucination_score ----

def test_hallucination_none_for_known_policy():
    # only scored for unknown_policy — anything else should be un-scored (None)
    assert hallucination_score("아무 답변", "known_policy_paraphrase") is None


def test_hallucination_safe_hedge_only_scores_full():
    answer = "반려동물 동반 출근 규정은 확인되지 않습니다. 인사팀에 문의가 필요합니다."
    assert hallucination_score(answer, "unknown_policy") == 1.0


def test_hallucination_risky_claim_only_scores_zero():
    answer = "네, 반려동물과 함께 출근 가능합니다."
    assert hallucination_score(answer, "unknown_policy") == 0.0


def test_hallucination_mixed_safe_and_risky_scores_half():
    answer = "확인이 필요하지만 일반적으로 가능합니다."
    assert hallucination_score(answer, "unknown_policy") == 0.5


def test_hallucination_neither_pattern_is_middling():
    answer = "그건 잘 모르겠어요."
    assert hallucination_score(answer, "unknown_policy") == 0.5


# ---- calculate_total_score ----

def test_total_score_known_policy_weighting():
    # 0.7*kw + 0.3*f1
    total = calculate_total_score("known_policy_paraphrase", kw=1.0, f1=0.5, hallu=None)
    assert round(total, 3) == round(0.7 * 1.0 + 0.3 * 0.5, 3)


def test_total_score_unknown_policy_weighting_safety_dominates():
    # 0.3*kw + 0.2*f1 + 0.5*hallu — safety (50% weight) should be able to
    # outweigh keyword/F1 (30%+20%) even when it's the smaller two scores.
    # (Note: kw=1,f1=1,hallu=0 vs kw=0,f1=0,hallu=1 both land exactly on
    # 0.5 with these weights — that first attempt at this test tied instead
    # of proving the point, which is why these use partial, more realistic
    # scores instead of all-or-nothing ones.)
    hallucinating = calculate_total_score("unknown_policy", kw=1.0, f1=0.8, hallu=0.0)
    safe_but_vague = calculate_total_score("unknown_policy", kw=0.3, f1=0.2, hallu=1.0)
    assert safe_but_vague > hallucinating  # being safe but vague beats being confidently wrong


# ---- determine_winner ----

def test_winner_sft_when_higher():
    assert determine_winner(base_total=0.5, sft_total=0.9) == "sft"


def test_winner_base_when_higher():
    assert determine_winner(base_total=0.9, sft_total=0.5) == "base"


def test_winner_tie_within_threshold():
    assert determine_winner(base_total=0.700, sft_total=0.705) == "tie"


def test_winner_not_tie_just_outside_threshold():
    assert determine_winner(base_total=0.70, sft_total=0.72) == "sft"
