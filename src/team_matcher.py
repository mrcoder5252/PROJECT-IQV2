"""
Project-IQ: Student Team Expertise Profiling & Domain Matcher
Evaluates student team skillsets, frameworks, and project interests against
project domains in the corpus, computing affinity scores, highlighting skill gaps,
and recommending the most suitable novel project ideas.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import config
from src.logger import get_logger

logger = get_logger("project_iq.team_matcher")

DOMAIN_TAXONOMY = {
    "Computer Vision & Healthcare Diagnostics": {
        "keywords": ["computer vision", "opencv", "cnn", "image processing", "pytorch", "medical", "disease detection", "yolo", "segmentation"],
        "description": "Medical imaging, patient monitoring, automated diagnosis from scans, and computer vision applied to healthcare."
    },
    "IoT, Edge AI & Smart Agriculture": {
        "keywords": ["iot", "sensors", "arduino", "raspberry pi", "edge computing", "agriculture", "soil", "crop", "embedded", "mqtt", "esp32"],
        "description": "Smart precision farming, environmental sensor networks, automated irrigation, and hardware edge inference."
    },
    "Cybersecurity, Privacy & Blockchain Systems": {
        "keywords": ["cryptography", "blockchain", "ethereum", "smart contract", "network security", "fraud detection", "intrusion detection", "solidity", "web3"],
        "description": "Decentralized ledgers, smart contracts, intrusion prevention, cryptographic verification, and privacy preservation."
    },
    "NLP, LLMs & Educational Technology": {
        "keywords": ["nlp", "transformers", "bert", "gpt", "huggingface", "text classification", "sentiment analysis", "rag", "education", "automated grading"],
        "description": "Natural language processing, educational assistants, automated summarization, and retrieval-augmented generation."
    },
    "Autonomous Mobility, Robotics & Smart Cities": {
        "keywords": ["robotics", "ros", "lidar", "autonomous vehicle", "traffic", "surveillance", "path planning", "drone", "slam"],
        "description": "Intelligent transportation systems, drone telemetry, robotic control, and smart city traffic optimization."
    },
    "Cloud Computing, Microservices & Data Engineering": {
        "keywords": ["docker", "kubernetes", "fastapi", "flask", "react", "postgresql", "kafka", "distributed systems", "microservices", "pipeline"],
        "description": "Scalable enterprise web applications, real-time distributed data pipelines, and cloud-native services."
    }
}


class TeamMatcher:
    def __init__(self, model_name: str = config.embedding_model_name):
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            logger.info(f"Loading SentenceTransformer for team matching: {model_name}")
            self.model = SentenceTransformer(model_name)

        self._domain_embeddings = {}
        for domain, info in DOMAIN_TAXONOMY.items():
            text = f"{domain}. {info['description']} Keywords: {', '.join(info['keywords'])}"
            self._domain_embeddings[domain] = self.model.encode(text)

    def evaluate_team(
        self,
        skills: List[str],
        interests: Optional[List[str]] = None,
        top_k: int = 3
    ) -> Dict[str, Any]:
        clean_skills = [s.strip() for s in skills if s and str(s).strip()]
        interests = [i.strip() for i in (interests or []) if i and str(i).strip()]

        if not clean_skills:
            logger.warning("Empty skills passed to team matcher.")
            unassessed_domains = [
                {
                    "domain": d,
                    "description": info["description"],
                    "match_percentage": 0.0,
                    "similarity_score": 0.0,
                    "matched_skills": [],
                    "suggested_to_learn": info["keywords"][:4],
                    "team_readiness": "Low"
                }
                for d, info in DOMAIN_TAXONOMY.items()
            ]
            return {
                "evaluated_skills": [],
                "evaluated_interests": interests,
                "top_domains": unassessed_domains[:top_k],
                "all_ranked_domains": unassessed_domains
            }

        team_desc = f"Team skills: {', '.join(clean_skills)}. Target interests: {', '.join(interests)}."
        team_embedding = self.model.encode(team_desc)

        ranked_domains = []
        for domain, domain_emb in self._domain_embeddings.items():
            # Cosine similarity
            dot = np.dot(team_embedding, domain_emb)
            norm = (np.linalg.norm(team_embedding) * np.linalg.norm(domain_emb)) + 1e-9
            sim = float(dot / norm)

            # Keyword overlap analysis
            info = DOMAIN_TAXONOMY[domain]
            matched_keywords = [
                k for k in info["keywords"]
                if any(k in s.lower() for s in clean_skills) or any(k in i.lower() for i in interests)
            ]
            missing_recommended_skills = [
                k for k in info["keywords"][:4]
                if k not in matched_keywords
            ]

            kw_coverage = min(1.0, len(matched_keywords) / 3.0) if matched_keywords else 0.0
            sim_clamped = max(0.0, sim)
            if kw_coverage > 0:
                match_percentage = min(100.0, max(0.0, round((sim_clamped * 0.4 + kw_coverage * 0.6) * 100, 1)))
            else:
                match_percentage = min(40.0, max(0.0, round(sim_clamped * 100 * 0.5, 1)))

            ranked_domains.append({
                "domain": domain,
                "description": info["description"],
                "match_percentage": match_percentage,
                "similarity_score": round(sim, 4),
                "matched_skills": matched_keywords,
                "suggested_to_learn": missing_recommended_skills,
                "team_readiness": "High" if match_percentage >= 70 else ("Moderate" if match_percentage >= 45 else "Low")
            })

        ranked_domains.sort(key=lambda x: x["match_percentage"], reverse=True)
        safe_k = max(1, min(top_k, len(ranked_domains)))

        return {
            "evaluated_skills": clean_skills,
            "evaluated_interests": interests,
            "top_domains": ranked_domains[:safe_k],
            "all_ranked_domains": ranked_domains
        }


def main():
    parser = argparse.ArgumentParser(description="Match student team skillsets to optimal project domains.")
    parser.add_argument("--skills", nargs="+", required=True, help="List of team skills (e.g. python pytorch opencv)")
    parser.add_argument("--interests", nargs="*", default=[], help="List of team interests (e.g. healthcare diagnostics)")
    args = parser.parse_args()

    matcher = TeamMatcher()
    result = matcher.evaluate_team(args.skills, args.interests)

    logger.info("================== TEAM MATCHING REPORT ==================")
    logger.info(f"Skills: {', '.join(result['evaluated_skills'])}")
    logger.info(f"Interests: {', '.join(result['evaluated_interests'])}")

    for i, d in enumerate(result["top_domains"], 1):
        logger.info(f"{i}. {d['domain']} — {d['match_percentage']}% Match [{d['team_readiness']} Readiness]")
        logger.info(f"   Matched: {', '.join(d['matched_skills']) if d['matched_skills'] else 'Semantic affinity'}")
        logger.info(f"   Recommended to learn: {', '.join(d['suggested_to_learn'])}")


if __name__ == "__main__":
    main()
