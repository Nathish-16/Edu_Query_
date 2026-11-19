import os
import tempfile
import unittest
from unittest.mock import patch

import backend.app as app_module
from backend.rag.retrieve import BM25Retriever, RAGDatabaseManager


class UploadIndexingTests(unittest.TestCase):
    def test_indexes_supported_uploads_missing_from_database(self):
        class FakeDatabase:
            def __init__(self):
                self.documents = [{"source": "existing.txt"}]
                self.added_chunks = []

            def list_documents(self):
                return self.documents

            def add_documents(self, chunks):
                self.added_chunks.extend(chunks)

        chunks = [{"text": "content", "metadata": {"source": "new.txt"}}]
        database = FakeDatabase()
        with tempfile.TemporaryDirectory() as upload_dir:
            for name in ("existing.txt", "new.txt", "ignore.bin"):
                open(os.path.join(upload_dir, name), "w").close()

            with patch.object(app_module, "UPLOADS_DIR", upload_dir), patch.object(
                app_module, "get_rag_db", return_value=database
            ), patch.object(
                app_module, "ingest_file", return_value=chunks
            ) as ingest:
                app_module._index_unindexed_uploads()

        ingest.assert_called_once_with(os.path.join(upload_dir, "new.txt"), "new.txt")
        self.assertEqual(database.added_chunks, chunks)

    def test_replaces_existing_chunks_for_same_source(self):
        class FakeStore:
            def __init__(self):
                self.operations = []

            def delete_document(self, source):
                self.operations.append(("delete", source))

            def add_documents(self, chunks, embeddings):
                self.operations.append(("add", chunks[0]["metadata"]["source"]))

            def get_all_chunks(self):
                return []

        database = RAGDatabaseManager.__new__(RAGDatabaseManager)
        database.db = FakeStore()
        database.bm25_retriever = BM25Retriever()
        chunks = [{"text": "updated", "metadata": {"source": "notes.txt"}}]

        with patch("backend.rag.retrieve.get_embeddings", return_value=[[0.1]]):
            database.add_documents(chunks)

        self.assertEqual(
            database.db.operations,
            [("delete", "notes.txt"), ("add", "notes.txt")],
        )

    def test_list_documents_includes_disk_files_when_index_is_empty(self):
        class FakeDatabase:
            def list_documents(self):
                return []

        with tempfile.TemporaryDirectory() as upload_dir:
            open(os.path.join(upload_dir, "lecture.pdf"), "w").close()
            with patch.object(app_module, "UPLOADS_DIR", upload_dir), patch.object(
                app_module, "get_rag_db", return_value=FakeDatabase()
            ):
                payload = app_module.list_documents()

        self.assertEqual(payload["documents"][0]["source"], "lecture.pdf")
        self.assertFalse(payload["documents"][0]["indexed"])

    def test_list_documents_does_not_wait_on_search_database(self):
        class BlockingDatabase:
            def list_documents(self):
                raise AssertionError("library listing must not query the search database")

        with tempfile.TemporaryDirectory() as upload_dir:
            open(os.path.join(upload_dir, "lecture.pdf"), "w").close()
            with patch.object(app_module, "UPLOADS_DIR", upload_dir), patch.object(
                app_module.rag_retrieve, "rag_db", BlockingDatabase()
            ), patch.object(app_module, "_INDEXED_CACHE", {"lecture.pdf": {"total_pages": 4}}):
                payload = app_module.list_documents()

        document = payload["documents"][0]
        self.assertEqual(document["source"], "lecture.pdf")
        self.assertEqual(document["total_pages"], 4)
        self.assertTrue(document["indexed"])


if __name__ == "__main__":
    unittest.main()
