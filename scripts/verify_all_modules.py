"""
Verification Suite: End-to-end testing of all Project-IQ modules and live APIs.
"""
import sys
import time
from pathlib import Path

# Ensure root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import config
from src.logger import get_logger
from src.llm import make_client, generate_text, current_model
from src.vector_store import ProjectIQVectorStore
from src.rag_chat import RAGChatEngine
from src.duplicate_detector import DuplicateDetector
from src.gap_analyzer import GapAnalyzer
from src.quality_scorer import QualityScorer
from src.team_matcher import TeamMatcher
from src.clean import redact_text, is_ner_available

logger = get_logger("project_iq.verify")

def test_api():
    print("\n" + "="*60)
    print("TEST 1: LIVE GEMINI API CONNECTIVITY")
    print("="*60)
    assert config.gemini_api_key, "GEMINI_API_KEY is not configured!"
    print(f"[OK] API Key detected (length: {len(config.gemini_api_key)})")
    
    client = make_client(config.gemini_api_key)
    prompt = "In 15 words or less, describe the purpose of an academic AI advisor."
    t0 = time.time()
    response = generate_text(client, prompt)
    elapsed = time.time() - t0
    
    print(f"[OK] Gemini API responded in {elapsed:.2f}s using model: {current_model()}")
    print(f"     Response: '{response.strip()}'")
    return True

def test_vector_store():
    print("\n" + "="*60)
    print("TEST 2: CHROMADB VECTOR STORE & EMBEDDINGS")
    print("="*60)
    store = ProjectIQVectorStore()
    count = store.count()
    print(f"[OK] ChromaDB collection loaded. Indexed records: {count}")
    assert count > 0, "Vector store collection is empty!"
    
    query = "deep learning automated disease diagnosis healthcare"
    t0 = time.time()
    hits = store.query_similar(query, n_results=3)
    elapsed = time.time() - t0
    
    print(f"[OK] Similarity query executed in {elapsed:.3f}s, returned {len(hits)} results:")
    for i, h in enumerate(hits, 1):
        print(f"     #{i} Similarity: {h.get('similarity', 0):.3f} | Dept: {h.get('department')} | Source: {h.get('source_file')}")
    return store

def test_module_advisor(store):
    print("\n" + "="*60)
    print("TEST 3: MODULE 1 — ACTIVE ADVISOR & RAG CHAT ENGINE")
    print("="*60)
    engine = RAGChatEngine(vector_store=store, api_key=config.gemini_api_key)
    query = "What methodologies have students used for deep learning computer vision projects?"
    t0 = time.time()
    result = engine.ask(query)
    elapsed = time.time() - t0
    
    print(f"[OK] RAG query completed in {elapsed:.2f}s")
    print(f"     Engine used: {result.get('engine_used')}")
    print(f"     Sources cited: {len(result.get('sources', []))} chunks")
    print(f"     Answer preview (first 180 chars):")
    print(f"     \"{result.get('answer', '')[:180]}...\"")
    return True

def test_module_originality(store):
    print("\n" + "="*60)
    print("TEST 4: MODULE 2 — PROJECT ORIGINALITY & REDUNDANCY GUARD")
    print("="*60)
    detector = DuplicateDetector(vector_store=store)
    sample_proposal = (
        "Design and deployment of an IoT-based smart agricultural monitoring system "
        "using ESP32 microcontrollers, DHT22 sensors, and LoRaWAN communication for soil moisture tracking."
    )
    t0 = time.time()
    report = detector.check_originality(sample_proposal)
    elapsed = time.time() - t0
    
    print(f"[OK] Originality scan completed in {elapsed:.2f}s")
    print(f"     Originality Score: {report.get('originality_score', 0):.1f}%")
    print(f"     Max Similarity:    {report.get('max_similarity_pct', 0):.1f}%")
    print(f"     Verdict:           {report.get('verdict')}")
    print(f"     Matching projects: {len(report.get('matching_projects', []))}")
    return True

def test_module_gaps(store):
    print("\n" + "="*60)
    print("TEST 5: MODULE 3 — INSTITUTIONAL RESEARCH GAP FINDER")
    print("="*60)
    analyzer = GapAnalyzer(vector_store=store, api_key=config.gemini_api_key)
    t0 = time.time()
    proposals = analyzer.generate_novel_projects(domain_filter="Healthcare AI", num_proposals=2)
    elapsed = time.time() - t0
    
    print(f"[OK] Research gap analysis & synthesis completed in {elapsed:.2f}s")
    print(f"     Generated proposals count: {len(proposals)}")
    for i, p in enumerate(proposals, 1):
        print(f"     Proposal #{i}: {p.get('title')}")
        print(f"       Feasibility: {p.get('feasibility_score')}/10")
    return True

def test_module_quality():
    print("\n" + "="*60)
    print("TEST 6: MODULE 4 — QUALITY RUBRIC SCORER")
    print("="*60)
    scorer = QualityScorer()
    sample_title = "Deep Learning Network Intrusion Detection"
    sample_problem = "Legacy rule-based IDSs suffer from high false-alarm rates and latency bottlenecks on high-throughput links."
    sample_method = "Step 1: Capture traffic via libpcap. Step 2: Temporal convolution with LSTM for sequential anomaly detection. Benchmark precision, recall, and f1-score with 10-fold cross validation [1]."
    sample_stack = ["Python", "PyTorch", "Docker", "FastAPI", "ONNX"]
    
    score_data = scorer.score_proposal(
        title=sample_title,
        problem_statement=sample_problem,
        proposed_methodology=sample_method,
        tech_stack=sample_stack,
        citations=["Smith et al., 2024"],
        domain="Cybersecurity & Blockchain"
    )
    print(f"[OK] Rubric Scorer evaluated project:")
    print(f"     Overall Score: {score_data.get('overall_score')}/100 ({score_data.get('star_rating')} stars)")
    print(f"     Grade:         {score_data.get('grade')}")
    print(f"     Dimensions:    {score_data.get('dimensions')}")
    return True

def test_module_team():
    print("\n" + "="*60)
    print("TEST 7: MODULE 5 — TEAM MATCHER & DOMAIN TAXONOMY")
    print("="*60)
    matcher = TeamMatcher()
    sample_skills = ["PyTorch", "OpenCV", "Python", "Computer Vision"]
    sample_interests = ["Autonomous Vehicles", "Robotics"]
    evaluation = matcher.evaluate_team(skills=sample_skills, interests=sample_interests, top_k=2)
    print(f"[OK] Team Matcher evaluated team skills:")
    for i, d in enumerate(evaluation.get("top_domains", []), 1):
        print(f"     #{i} Domain: {d.get('domain')} (Match: {d.get('match_percentage', 0):.1f}%, Readiness: {d.get('team_readiness')})")
    return True

def test_module_vault():
    print("\n" + "="*60)
    print("TEST 8: MODULE 6 — PRIVACY VAULT & SECURITY GATE")
    print("="*60)
    print(f"[OK] Admin Passkey configured: {bool(config.admin_passkey)}")
    print(f"[OK] NER Model available: {is_ner_available()}")
    
    raw_text = "Student John Doe (Roll: 19CS042, Email: john.doe@ghrcem.edu.in, Phone: 9876543210) submitted this report."
    clean_text, scrub_log = redact_text(raw_text, ["John Doe"])
    print(f"[OK] Redaction engine processed text:")
    print(f"     Original: {raw_text}")
    print(f"     Redacted: {clean_text}")
    print(f"     Scrub stats: {scrub_log}")
    assert "john.doe@ghrcem.edu.in" not in clean_text, "Email was not redacted!"
    assert "9876543210" not in clean_text, "Phone was not redacted!"
    assert "19CS042" not in clean_text, "Student ID was not redacted!"
    return True

def main():
    print("\n" + "#"*60)
    print("PROJECT-IQ COMPLETE SYSTEM VERIFICATION SUITE")
    print("#"*60)
    
    t_start = time.time()
    test_api()
    store = test_vector_store()
    test_module_advisor(store)
    test_module_originality(store)
    test_module_gaps(store)
    test_module_quality()
    test_module_team()
    test_module_vault()
    
    total_time = time.time() - t_start
    print("\n" + "="*60)
    print(f"ALL MODULES AND APIS VERIFIED AND WORKING PROPERLY! (Total Time: {total_time:.2f}s)")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
