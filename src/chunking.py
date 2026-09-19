from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        text = text.strip()
        if not text:
            return []

        raw_sentences = re.split(r'(?<=[.!?])(?:\s+|\n+)', text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        if not sentences:
            return []

        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(group))
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return []
        if len(current_text) <= self.chunk_size:
            return [current_text]
        if not remaining_separators:
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]

        if separator == "":
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        if separator not in current_text:
            return self._split(current_text, next_separators)

        splits = current_text.split(separator)
        sub_chunks: list[str] = []
        for part in splits:
            if not part:
                continue
            if len(part) > self.chunk_size:
                sub_chunks.extend(self._split(part, next_separators))
            else:
                sub_chunks.append(part)

        merged_chunks: list[str] = []
        current_chunk = ""
        for piece in sub_chunks:
            if not current_chunk:
                current_chunk = piece
            else:
                candidate = current_chunk + separator + piece
                if len(candidate) <= self.chunk_size:
                    current_chunk = candidate
                else:
                    merged_chunks.append(current_chunk)
                    current_chunk = piece

        if current_chunk:
            merged_chunks.append(current_chunk)

        return merged_chunks


class HeadingChunker:
    """
    Chunk markdown documents based on section headings (e.g. #, ##, ###, Điều ...).

    Features:
        - Splits text by markdown headings (#, ##, ###, ...) or regulation articles ('Điều X').
        - Keeps heading context intact with each section.
        - If a section exceeds max_chunk_size, recursively splits it using RecursiveChunker
          and re-attaches the section heading to every sub-chunk so context is never lost.
    """

    def __init__(self, max_chunk_size: int = 500, heading_pattern: str | None = None) -> None:
        self.max_chunk_size = max_chunk_size
        self.heading_pattern = heading_pattern or r'(?m)^(?=#{1,6}\s+|Điều\s+\d+)'
        self._fallback_chunker = RecursiveChunker(chunk_size=max_chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        text = text.strip()
        if not text:
            return []

        sections = [s.strip() for s in re.split(self.heading_pattern, text) if s.strip()]
        if not sections:
            return []

        chunks: list[str] = []
        for section in sections:
            if len(section) <= self.max_chunk_size:
                chunks.append(section)
            else:
                lines = section.splitlines()
                heading_line = lines[0].strip() if lines else ""
                body = "\n".join(lines[1:]).strip() if len(lines) > 1 else section

                sub_chunks = self._fallback_chunker.chunk(body)
                for sub in sub_chunks:
                    if heading_line and not sub.startswith(heading_line):
                        chunks.append(f"{heading_line}\n{sub}")
                    else:
                        chunks.append(sub)

        return chunks


def _dot(a: list[float], b: list[float]) -> float:

    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(y * y for y in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=min(50, chunk_size // 10)),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }

        comparison = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text) if text else []
            count = len(chunks)
            avg_len = sum(len(c) for c in chunks) / count if count > 0 else 0.0
            comparison[name] = {
                "count": count,
                "avg_length": avg_len,
                "chunks": chunks,
            }
        return comparison

