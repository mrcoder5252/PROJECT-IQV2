"""
Unit tests for Project-IQ RAG Chat Engine (src/rag_chat.py)
Validates:
1. RAGChatEngine initialization and configuration.
2. Vector retrieval filtering across departments and sections.
3. Offline local structured synthesis and citation formatting.
4. Empty query handling and edge cases.
5. Dynamic API key configuration.
"""

import unittest
from unittest.mock import MagicMock, patch
from src.rag_chat import RAGChatEngine, RAG_SYSTEM_PROMPT


class TestRAGChatEngine(unittest.TestCase):
    def setUp(self):
        # Create a mock vector store to isolate tests from disk/network
        self.mock_vector_store = MagicMock()
        self.mock_chunks = [
            {
                "text": "The proposed system uses a Convolutional Neural Network (ResNet50) for plant leaf disease detection, achieving 94.2% accuracy.",
                "similarity": 0.88,
                "metadata": {
                    "source_file": "Plant_Disease_CS1_G04.docx",
                    "department": "CS_1",
                    "group_id": "Group_04",
                    "section": "METHODOLOGY",
                    "category": "blackbook",
                    "token_count": 85,
                    "has_citation": True,
                    "has_numbers": True
                }
            },
            {
                "text": "Experimental evaluation demonstrated inference latency of 42ms on desktop GPU, but mobile deployment was constrained by memory limits.",
                "similarity": 0.82,
                "metadata": {
                    "source_file": "Plant_Disease_CS1_G04.docx",
                    "department": "CS_1",
                    "group_id": "Group_04",
                    "section": "RESULTS",
                    "category": "blackbook",
                    "token_count": 92,
                    "has_citation": False,
                    "has_numbers": True
                }
            },
            {
                "text": "Future scope includes quantizing the model to INT8 precision for edge devices and expanding the dataset to include rare crop pathologies.",
                "similarity": 0.79,
                "metadata": {
                    "source_file": "Plant_Disease_CS1_G04.docx",
                    "department": "CS_1",
                    "group_id": "Group_04",
                    "section": "FUTURE SCOPE",
                    "category": "blackbook",
                    "token_count": 65,
                    "has_citation": False,
                    "has_numbers": False
                }
            }
        ]
        self.mock_vector_store.query_similar.return_value = self.mock_chunks

    def test_engine_init(self):
        """Engine initializes with default or custom vector store and detects API key."""
        engine = RAGChatEngine(vector_store=self.mock_vector_store, api_key=None)
        self.assertIsNotNone(engine.vector_store)
        self.assertIsNone(engine._gemini_client)

    def test_retrieve_context(self):
        """Context retrieval calls vector store and filters appropriately."""
        engine = RAGChatEngine(vector_store=self.mock_vector_store)
        chunks = engine.retrieve_context("plant disease detection", n_results=3)
        self.assertEqual(len(chunks), 3)
        self.mock_vector_store.query_similar.assert_called_once()

    def test_retrieve_context_dept_filter(self):
        """Department filter strictly retains only matching chunks."""
        engine = RAGChatEngine(vector_store=self.mock_vector_store)
        chunks = engine.retrieve_context("plant disease", n_results=3, department_filter="CS_1")
        self.assertTrue(all(c["metadata"]["department"] == "CS_1" for c in chunks))

    def test_local_synthesis_structure(self):
        """Local synthesis produces grounded academic response with citations and sections."""
        engine = RAGChatEngine(vector_store=self.mock_vector_store)
        res = engine.ask("What models and limitations were reported for plant disease detection?")
        
        self.assertIn("answer", res)
        self.assertIn("sources", res)
        self.assertIn("engine_used", res)
        self.assertEqual(res["engine_used"], "Local High-Fidelity RAG Synthesizer")
        
        answer = res["answer"]
        self.assertIn("Academic Synthesis", answer)
        self.assertIn("Plant_Disease_CS1_G04.docx", answer)
        self.assertIn("Strategic Guidance", answer)

    def test_citations_content(self):
        """Returned sources contain complete metadata and excerpts."""
        engine = RAGChatEngine(vector_store=self.mock_vector_store)
        res = engine.ask("Tell me about inference latency")
        sources = res["sources"]
        
        self.assertEqual(len(sources), 3)
        s1 = sources[0]
        self.assertEqual(s1["source_file"], "Plant_Disease_CS1_G04.docx")
        self.assertEqual(s1["department"], "CS_1")
        self.assertEqual(s1["group_id"], "Group_04")
        self.assertEqual(s1["section"], "METHODOLOGY")
        self.assertGreater(s1["similarity_score"], 0.0)
        self.assertTrue(len(s1["excerpt"]) > 10)

    def test_empty_query_handling(self):
        """Handles empty or non-matching retrieval gracefully without throwing errors."""
        self.mock_vector_store.query_similar.return_value = []
        engine = RAGChatEngine(vector_store=self.mock_vector_store)
        res = engine.ask("xyznonexistentterm123")
        
        self.assertEqual(res["engine_used"], "no_context")
        self.assertEqual(len(res["sources"]), 0)
        self.assertIn("could not find any relevant project documents", res["answer"].lower())

    def test_dynamic_api_key_update(self):
        """Engine supports updating API key at runtime."""
        engine = RAGChatEngine(vector_store=self.mock_vector_store)
        with patch("google.genai.Client") as mock_client:
            engine.set_api_key("test-gemini-key")
            self.assertEqual(engine.api_key, "test-gemini-key")


if __name__ == "__main__":
    unittest.main()
