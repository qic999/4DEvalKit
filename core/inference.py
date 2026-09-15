"""OpenAI-compatible text completion client and offline replay; no GPU imports."""
from __future__ import annotations

import json
import http.client
import os
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .io import read_json


class APIInferenceEngine:
    def __init__(self, *, model, base_url, api_key_env="OPENAI_API_KEY", timeout=120,
                 retries=2, max_tokens=512, temperature=0.0, seed=0, extra_body=None):
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("--base-url must be an HTTP(S) server URL without embedded credentials or query parameters")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.timeout, self.retries = model, timeout, retries
        self.api_key = os.getenv(api_key_env, "")
        self.generation = dict(max_tokens=max_tokens, temperature=temperature, seed=seed)
        self.extra_body = extra_body or {}
        if {"model", "messages", "stream"} & self.extra_body.keys():
            raise ValueError("extra_body cannot replace model/messages/stream")

    def infer(self, messages, *, structured_outputs=None):
        payload = {"model": self.model, "messages": messages, **self.generation, **self.extra_body}
        if structured_outputs is not None:
            if 'structured_outputs' in self.extra_body or 'response_format' in self.extra_body:
                raise ValueError('Per-question answer constraints conflict with extra_body')
            payload['structured_outputs'] = structured_outputs
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        start = time.monotonic()
        for attempt in range(self.retries + 1):
            try:
                request = urllib.request.Request(self.url, data=json.dumps(payload).encode(), headers=headers)
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.load(response)
                choice = data["choices"][0]
                text = choice["message"].get("content")
                if not isinstance(text, str):
                    raise ValueError("Completion has no final text content")
                finish = choice.get("finish_reason")
                status = "ok" if finish in {None, "stop", "eos_token"} and text.strip() else "invalid_completion"
                if structured_outputs is not None and status == 'ok':
                    from .response_constraints import matches_constraint
                    if not matches_constraint(text, structured_outputs):
                        status = 'invalid_completion'
                return {"raw_output": text, "status": status, "finish_reason": finish,
                        "usage": data.get("usage", {}), "latency_seconds": time.monotonic() - start}
            except (OSError, http.client.HTTPException, ValueError, KeyError, IndexError, TypeError) as exc:
                retryable = not isinstance(exc, urllib.error.HTTPError) or exc.code in {408, 429, 500, 502, 503, 504}
                if attempt < self.retries and retryable:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                # Do not write response bodies or authorization values into logs.
                return {"raw_output": "", "status": "error", "error_type": type(exc).__name__,
                        "http_status": exc.code if isinstance(exc, urllib.error.HTTPError) else None,
                        "latency_seconds": time.monotonic() - start}


class ReplayEngine:
    def __init__(self, path):
        source = read_json(path)
        if isinstance(source, dict) and "results" in source:
            source = source["results"]
        if isinstance(source, dict):
            source = [{"sample_id": key, "raw_output": value} if isinstance(value, (str, int, float))
                      else {"sample_id": key, **value} for key,value in source.items()]
        if not isinstance(source, list):
            raise ValueError("Predictions must be a result bundle, list, JSONL, or ID-to-output map")
        self.outputs = {}
        for row in source:
            key = row.get("sample_id", row.get("id", row.get("question_id")))
            if key is None:
                raise ValueError("Prediction row has no sample_id/id/question_id")
            legacy_vsi = ("qa_index" in row and str(row.get("id")) == f"vsibench_{row['qa_index']}")
            if legacy_vsi:
                # Existing eval.py uses QA ARRAY INDEX, not the native id field.
                key = f"vsi_bench:{row['qa_index']}"
            output = next((row[name] for name in ["raw_output", "pred_answer", "prediction"] if name in row), None)
            if not isinstance(output, str):
                if isinstance(output, (int, float)):
                    output = str(output)
                else:
                    raise ValueError(f"Prediction {key} needs raw_output/pred_answer/prediction")
            if str(key) in self.outputs:
                raise ValueError(f"Duplicate prediction ID: {key}")
            status = row.get("status", "ok")
            if status == "ok" and (row.get("finish_reason") not in {None, "stop", "eos_token"} or not output.strip()):
                status = "invalid_completion"
            self.outputs[str(key)] = {"raw_output": output, "status": status,
                                      "finish_reason": row.get("finish_reason"), "replayed": True}
            if legacy_vsi:
                self.outputs[str(key)]["legacy_input_identity"] = {"question": row.get("question"),
                    "scene_name": row.get("scene_name"), "options": row.get("options")}

    def infer(self, sample_id, source_id, *, question=None, metadata=None):
        for key in [sample_id, source_id]:
            if key is not None and str(key) in self.outputs:
                result = self.outputs[str(key)]
                identity = result.get("legacy_input_identity")
                if identity:
                    old_question = (identity.get("question") or "").strip()
                    meta = metadata or {}
                    if (not question or not old_question or
                        not (question.strip() == old_question or question.strip().startswith(old_question + "\n")) or
                        identity.get("scene_name") != meta.get("scene_name") or
                        identity.get("options") != meta.get("options")):
                        raise ValueError(f"Legacy VSI QA index no longer matches question/scene/options: {sample_id}")
                return result
        return {"raw_output": "", "status": "missing_prediction", "replayed": True}
