"""
chunker.py
==========
Document loading and chunking pipeline.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


class DocumentChunker:
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def chunk_file(self, file_path, metadata_overrides=None, chunk_size=512, chunk_overlap=50):
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            raw_chunks = self._chunk_pdf(file_path, chunk_size, chunk_overlap)
        elif suffix == ".md":
            raw_chunks = self._chunk_markdown(file_path, chunk_size, chunk_overlap)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        metadata = self._infer_metadata(file_path, metadata_overrides)
        chunks = []
        for raw in raw_chunks:
            text = raw["text"].strip()
            if len(text) < 50:
                continue
            chunk_id = VectorStoreManager.generate_chunk_id(file_path.name, text)
            chunks.append(DocumentChunk(chunk_id=chunk_id, chunk_text=text, metadata=metadata))
        logger.info(f"Chunked {file_path.name} into {len(chunks)} chunks")
        return chunks

    def chunk_files(self, file_paths, metadata_overrides=None):
        all_chunks = []
        for file_path in file_paths:
            try:
                chunks = self.chunk_file(file_path, metadata_overrides)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to chunk {file_path.name}: {e}")
        return all_chunks

    def _chunk_pdf(self, file_path, chunk_size, chunk_overlap):
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = splitter.split_documents(pages)
        return [{"text": doc.page_content} for doc in docs]

    def _chunk_markdown(self, file_path, chunk_size, chunk_overlap):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        content = file_path.read_text(encoding="utf-8")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", " "],
        )
        texts = splitter.split_text(content)
        return [{"text": t} for t in texts]

    def _infer_metadata(self, file_path, overrides=None):
        bonus_topics = {"som", "boltzmann", "gan"}
        stem = file_path.stem.lower()
        parts = stem.split("_")
        topic = parts[0].upper() if parts else "UNKNOWN"
        difficulty = parts[1] if len(parts) > 1 else "intermediate"
        is_bonus = topic.lower() in bonus_topics
        meta = ChunkMetadata(
            topic=topic,
            difficulty=difficulty,
            type="concept_explanation",
            source=file_path.name,
            is_bonus=is_bonus,
        )
        if overrides:
            for key, value in overrides.items():
                if hasattr(meta, key):
                    setattr(meta, key, value)
        return meta
