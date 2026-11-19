import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from backend.rag.ingest import extract_pdf_chunks, extract_pptx_chunks, ingest_file
from backend.rag.retrieve import get_rag_db, get_embeddings
from backend.app import (
    _index_saved_file,
    _index_unindexed_uploads,
    list_documents,
    get_status,
    UPLOADS_DIR,
)


class TestEndToEndChromaUpload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Make the suite order-independent: every test expects the sample
        # uploads to already be present in ChromaDB and the index cache.
        _index_unindexed_uploads()

    def setUp(self):
        self.db = get_rag_db()

    def test_pdf_extraction_and_indexing(self):
        pdf_path = os.path.join(UPLOADS_DIR, "Unit 4.3 Testing Services.pdf")
        self.assertTrue(os.path.isfile(pdf_path), "Sample PDF must exist in uploads")

        chunks = extract_pdf_chunks(pdf_path, "Unit 4.3 Testing Services.pdf")
        self.assertGreater(len(chunks), 0, "PDF should yield at least 1 chunk")
        self.assertEqual(chunks[0]["metadata"]["type"], "pdf")
        self.assertGreater(chunks[0]["metadata"]["total_pages"], 0)

        # Test indexing into ChromaDB
        indexed_count = _index_saved_file("Unit 4.3 Testing Services.pdf")
        self.assertEqual(indexed_count, len(chunks))

        # Check ChromaDB has the document
        docs = self.db.list_documents()
        sources = [d["source"] for d in docs]
        self.assertIn("Unit 4.3 Testing Services.pdf", sources)

    def test_pptx_extraction_and_indexing(self):
        pptx_path = os.path.join(UPLOADS_DIR, "sample_lecture.pptx")
        self.assertTrue(os.path.isfile(pptx_path), "Sample PPTX must exist in uploads")

        chunks = extract_pptx_chunks(pptx_path, "sample_lecture.pptx")
        self.assertGreater(len(chunks), 0, "PPTX should yield at least 1 chunk")
        self.assertEqual(chunks[0]["metadata"]["type"], "pptx")
        self.assertGreater(chunks[0]["metadata"]["slide_count"], 0)

        # Test indexing into ChromaDB
        indexed_count = _index_saved_file("sample_lecture.pptx")
        self.assertEqual(indexed_count, len(chunks))

        # Check ChromaDB has the document
        docs = self.db.list_documents()
        sources = [d["source"] for d in docs]
        self.assertIn("sample_lecture.pptx", sources)

    def test_chromadb_status(self):
        status = get_status()
        self.assertEqual(status["active_db"], "chromadb")
        self.assertEqual(status["status"], "healthy")

    def test_chromadb_query_retrieval(self):
        # Query for testing services (from PDF)
        results = self.db.query("testing services", n_results=3)
        self.assertGreater(len(results), 0, "Should retrieve chunks for PDF query")

        # Query for algorithm (from PPTX)
        results_pptx = self.db.query("algorithm step by step", n_results=3)
        self.assertGreater(len(results_pptx), 0, "Should retrieve chunks for PPTX query")

    def test_list_documents_endpoint(self):
        doc_list = list_documents()
        self.assertIn("documents", doc_list)
        sources = {d["source"]: d for d in doc_list["documents"]}

        self.assertIn("Unit 4.3 Testing Services.pdf", sources)
        self.assertTrue(sources["Unit 4.3 Testing Services.pdf"]["indexed"])
        self.assertGreater(sources["Unit 4.3 Testing Services.pdf"].get("total_pages", 0), 0)

        self.assertIn("sample_lecture.pptx", sources)
        self.assertTrue(sources["sample_lecture.pptx"]["indexed"])
        self.assertGreater(sources["sample_lecture.pptx"].get("slide_count", 0), 0)


if __name__ == "__main__":
    unittest.main()
