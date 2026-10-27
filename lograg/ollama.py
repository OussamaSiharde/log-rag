"""Minimal client for a local Ollama server (embeddings + chat). No data leaves the host."""

import json
from typing import Callable, Protocol, Sequence

import httpx

# nomic-embed-text is trained with task prefixes; using them measurably improves retrieval.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class ChatModel(Protocol):
    def chat(self, system: str, user: str, on_token: Callable[[str], None] | None = None) -> str: ...


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str, embed_model: str, chat_model: str, timeout: float = 600.0):
        self._http = httpx.Client(base_url=base_url, timeout=timeout)
        self.embed_model = embed_model
        self.chat_model = chat_model

    def _post(self, path: str, payload: dict) -> dict:
        try:
            resp = self._http.post(path, json=payload)
        except httpx.ConnectError as exc:
            raise OllamaError(f"Ollama is not reachable at {self._http.base_url} ({exc}). Is it running?") from exc
        if resp.status_code == 404 and "not found" in resp.text:
            model = payload.get("model")
            raise OllamaError(f"Model '{model}' is not pulled. Run: ollama pull {model}")
        resp.raise_for_status()
        return resp.json()

    def _embed(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            vectors.extend(self._post("/api/embed", {"model": self.embed_model, "input": batch})["embeddings"])
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed([DOC_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([QUERY_PREFIX + text])[0]

    def chat(self, system: str, user: str, on_token: Callable[[str], None] | None = None) -> str:
        """Stream the answer: on a CPU-only machine a 7B model can take minutes to read the
        prompt, and streaming means no single silent read that a timeout could cut off."""
        payload = {
            "model": self.chat_model,
            "stream": True,
            # Low temperature: the answer should restate the logs, not get creative about them.
            "options": {"temperature": 0.1, "num_ctx": 8192},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        parts: list[str] = []
        try:
            # No read timeout: the first token may legitimately take minutes on CPU.
            with self._http.stream("POST", "/api/chat", json=payload, timeout=httpx.Timeout(10.0, read=None)) as resp:
                if resp.status_code == 404:
                    raise OllamaError(f"Model '{self.chat_model}' is not pulled. Run: ollama pull {self.chat_model}")
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise OllamaError(chunk["error"])
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        parts.append(token)
                        if on_token:
                            on_token(token)
                    if chunk.get("done"):
                        break
        except httpx.ConnectError as exc:
            raise OllamaError(f"Ollama is not reachable at {self._http.base_url} ({exc}). Is it running?") from exc
        return "".join(parts).strip()
