"""Regression tests for the bugs fixed in the cleanup pass (no ML models needed)."""
import os
import re

from src.config import Config
from src.extract import parse_college_hierarchy
from src.quality_scorer import QualityScorer
from src.clean import is_admin_front_matter, redact_text, looks_like_name
from src.gap_analyzer import compute_feasibility_score
from src.rag_chat import is_off_topic_query


def test_gemini_key_is_read_live(monkeypatch):
    cfg = Config()
    monkeypatch.setenv("GEMINI_API_KEY", "first")
    assert cfg.gemini_api_key == "first"
    monkeypatch.setenv("GEMINI_API_KEY", "second")
    assert cfg.gemini_api_key == "second"
    monkeypatch.delenv("GEMINI_API_KEY")
    assert cfg.gemini_api_key is None


def test_legacy_env_prefix_and_no_default_passkey(monkeypatch):
    monkeypatch.delenv("PROJECTIQ_ADMIN_PASSKEY", raising=False)
    monkeypatch.delenv("PROJECT_IQ_ADMIN_PASSKEY", raising=False)
    assert Config().admin_passkey is None
    monkeypatch.delenv("PROJECTIQ_LLM_MODEL", raising=False)
    monkeypatch.setenv("PROJECT_IQ_LLM_MODEL", "legacy-model")
    assert Config().llm_model_name == "legacy-model"


def test_hierarchy_variants():
    for rel, dept, grp in [
        ("DS_1/Group 9 document/blackbook.pdf", "DS_1", "Group 9"),
        ("DS1/Group_3/x.pdf", "DS_1", "Group 3"),
        ("CS-2/group 12/x.pdf", "CS_2", "Group 12"),
    ]:
        m = parse_college_hierarchy(rel)
        assert (m["department"], m["group_id"]) == (dept, grp)


def test_category_folder_beats_filename():
    assert parse_college_hierarchy("DS_1/Group 1/Review 4/final_report.pdf")["category"] == "review4"
    assert parse_college_hierarchy("DS_1/Group 1/newspaper.pdf")["category"] == "project_report"


def test_percentages_and_steps_are_scored():
    s = QualityScorer()
    f = s.extract_features("t", "problem", "1. ingest\n2. train\n", ["python"], None, None,
                           "We reach 95% accuracy, 0.91 F1 score and 40 ms latency.")
    assert f["metric_rigor"] >= 75
    assert f["methodology_depth"] >= 45


def test_repeating_a_metric_does_not_inflate_score():
    s = QualityScorer()
    f = s.extract_features("t", "p", "accuracy " * 20, [], None, None, None)
    assert f["metric_rigor"] <= 25


def test_domain_bonus_not_circular():
    s = QualityScorer()
    plain = s.extract_features("t", "p", "m", ["python"], None, None, None)["tech_stack_coherence"]
    domain_only = s.extract_features("t", "p", "m", ["python"], None, "Computer Vision", None)["tech_stack_coherence"]
    assert plain == domain_only  # label alone gives no credit


def test_admin_phrase_only_at_block_start():
    assert is_admin_front_matter("This is to certify that the student ...")
    assert not is_admin_front_matter("The receiver sends an acknowledgement after every packet in the TCP handshake " * 3)


def test_decimal_series_not_redacted_as_phone():
    text, counts = redact_text("Scores 0.95 0.93 0.91 0.89 0.87 0.85 and call +91 98765 43210", [])
    assert "0.95 0.93 0.91 0.89" in text and counts["phone"] == 1


def test_titles_are_not_learned_as_names():
    assert not looks_like_name("PLANT DISEASE DETECTION")
    assert not looks_like_name("Smart Attendance System")
    assert looks_like_name("Rahul Sharma")


def test_feasibility_not_constant():
    assert compute_feasibility_score([]) != compute_feasibility_score(["latency", "real-time", "limitation"], 0.5, True)
    assert 3 <= compute_feasibility_score([]) <= 10


def test_guardrail_generic_words_do_not_bypass():
    assert is_off_topic_query("give me a pasta recipe for my project system data dinner")
    assert not is_off_topic_query("what dataset and algorithm for plant disease detection")
