"""Model runners — adapters for different LLM providers."""

import time
from abc import ABC, abstractmethod

import httpx
from openai import OpenAI

from .models import Task, TaskResult

SYSTEM_PROMPT = "Du bist ein hilfreicher Assistent. Antworte präzise und fachlich korrekt auf Deutsch."


class ModelRunner(ABC):
    """Base class for model adapters."""

    @abstractmethod
    def run(self, task: Task) -> TaskResult:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...


class OpenAIRunner(ModelRunner):
    """Runner for OpenAI API (GPT-4o, GPT-4o-mini, etc.)."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        self._model = model
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    @property
    def model_name(self) -> str:
        return f"openai/{self._model}"

    def run(self, task: Task) -> TaskResult:
        start = time.perf_counter_ns()

        # Reasoning models (gpt-5*, o-series) don't accept temperature and need large token budgets
        is_reasoning = any(x in self._model for x in ("gpt-5", "o1", "o3", "o4"))
        is_new_api = any(x in self._model for x in ("gpt-5", "gpt-4.1", "o1", "o3", "o4"))

        kwargs: dict = {}
        if is_new_api:
            # Reasoning models need large budgets (reasoning tokens + output tokens)
            kwargs["max_completion_tokens"] = 16000 if is_reasoning else 2048
        else:
            kwargs["max_tokens"] = 2048

        if not is_reasoning:
            kwargs["temperature"] = 0.0

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task.prompt},
            ],
            **kwargs,
        )
        latency_ms = (time.perf_counter_ns() - start) // 1_000_000

        choice = response.choices[0]
        usage = response.usage

        return TaskResult(
            task_id=task.id,
            model=self.model_name,
            response=choice.message.content or "",
            score=0.0,
            latency_ms=latency_ms,
            tokens_in=usage.prompt_tokens if usage else 0,
            tokens_out=usage.completion_tokens if usage else 0,
        )


class OllamaRunner(ModelRunner):
    """Runner for local Ollama models."""

    def __init__(self, model: str = "llama3:8b", base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = base_url

    @property
    def model_name(self) -> str:
        return f"ollama/{self._model}"

    def run(self, task: Task) -> TaskResult:
        start = time.perf_counter_ns()
        resp = httpx.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task.prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=120.0,
        )
        latency_ms = (time.perf_counter_ns() - start) // 1_000_000
        data = resp.json()

        return TaskResult(
            task_id=task.id,
            model=self.model_name,
            response=data.get("message", {}).get("content", ""),
            score=0.0,
            latency_ms=latency_ms,
            tokens_in=data.get("prompt_eval_count", 0),
            tokens_out=data.get("eval_count", 0),
        )


class BedrockRunner(ModelRunner):
    """Runner for Amazon Bedrock (Claude, Llama, etc.)."""

    def __init__(self, model_id: str = "anthropic.claude-sonnet-4-20250514-v1:0", region: str = "eu-west-1"):
        import boto3
        self._model_id = model_id
        self._region = region
        self._client = boto3.client("bedrock-runtime", region_name=region)

    @property
    def model_name(self) -> str:
        # Shorten model ID for display
        short = self._model_id.split(".")[1] if "." in self._model_id else self._model_id
        return f"bedrock/{short}"

    def run(self, task: Task) -> TaskResult:
        import json

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": task.prompt},
            ],
        }

        # Sonnet 5+ and Opus 4.7+ don't accept temperature (adaptive thinking)
        if not any(x in self._model_id for x in ("sonnet-5", "opus-4-7", "opus-4-8")):
            body["temperature"] = 0.0

        start = time.perf_counter_ns()
        response = self._client.invoke_model(
            modelId=self._model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        latency_ms = (time.perf_counter_ns() - start) // 1_000_000

        result = json.loads(response["body"].read())
        content = result.get("content", [{}])[0].get("text", "")
        usage = result.get("usage", {})

        return TaskResult(
            task_id=task.id,
            model=self.model_name,
            response=content,
            score=0.0,
            latency_ms=latency_ms,
            tokens_in=usage.get("input_tokens", 0),
            tokens_out=usage.get("output_tokens", 0),
        )


class BedrockMistralRunner(ModelRunner):
    """Runner for Mistral models on Bedrock."""

    def __init__(self, model_id: str = "mistral.mistral-large-2402-v1:0", region: str = "eu-west-1"):
        import boto3
        self._model_id = model_id
        self._region = region
        self._client = boto3.client("bedrock-runtime", region_name=region)

    @property
    def model_name(self) -> str:
        short = self._model_id.replace("mistral.", "").split(":")[0]
        return f"bedrock/{short}"

    def run(self, task: Task) -> TaskResult:
        import json

        body = {
            "max_tokens": 2048,
            "temperature": 0.0,
            "messages": [
                {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{task.prompt}"},
            ],
        }

        start = time.perf_counter_ns()
        response = self._client.invoke_model(
            modelId=self._model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        latency_ms = (time.perf_counter_ns() - start) // 1_000_000

        result = json.loads(response["body"].read())
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = result.get("usage", {})

        return TaskResult(
            task_id=task.id,
            model=self.model_name,
            response=content,
            score=0.0,
            latency_ms=latency_ms,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
        )


class BedrockNovaRunner(ModelRunner):
    """Runner for Amazon Nova models (different API format)."""

    def __init__(self, model_id: str = "amazon.nova-pro-v1:0", region: str = "eu-west-1"):
        import boto3
        self._model_id = model_id
        self._region = region
        self._client = boto3.client("bedrock-runtime", region_name=region)

    @property
    def model_name(self) -> str:
        short = self._model_id.replace("amazon.", "").split(":")[0]
        return f"bedrock/{short}"

    def run(self, task: Task) -> TaskResult:
        import json

        body = {
            "inferenceConfig": {"maxTokens": 2048, "temperature": 0.0},
            "system": [{"text": SYSTEM_PROMPT}],
            "messages": [
                {"role": "user", "content": [{"text": task.prompt}]},
            ],
        }

        start = time.perf_counter_ns()
        response = self._client.invoke_model(
            modelId=self._model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        latency_ms = (time.perf_counter_ns() - start) // 1_000_000

        result = json.loads(response["body"].read())
        content = result.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")
        usage = result.get("usage", {})

        return TaskResult(
            task_id=task.id,
            model=self.model_name,
            response=content,
            score=0.0,
            latency_ms=latency_ms,
            tokens_in=usage.get("inputTokens", 0),
            tokens_out=usage.get("outputTokens", 0),
        )


class BedrockMantleRunner(ModelRunner):
    """Runner for OpenAI models on Bedrock via the Mantle endpoint (Responses API + SigV4)."""

    def __init__(self, model_id: str = "openai.gpt-5.6-sol", region: str = "us-east-1"):
        import boto3
        import requests
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        self._model_id = model_id
        self._region = region
        self._session = boto3.Session(region_name=region)
        self._url = f"https://bedrock-mantle.{region}.api.aws/openai/v1/responses"

    @property
    def model_name(self) -> str:
        short = self._model_id.replace("openai.", "")
        return f"bedrock-mantle/{short}"

    def run(self, task: Task) -> TaskResult:
        import json
        import requests
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        credentials = self._session.get_credentials().get_frozen_credentials()
        body = json.dumps({
            "model": self._model_id,
            "instructions": SYSTEM_PROMPT,
            "input": task.prompt,
        })

        aws_request = AWSRequest(method="POST", url=self._url, data=body, headers={"Content-Type": "application/json"})
        SigV4Auth(credentials, "bedrock", self._region).add_auth(aws_request)

        start = time.perf_counter_ns()
        r = requests.post(self._url, headers=dict(aws_request.headers), data=body, timeout=300)
        latency_ms = (time.perf_counter_ns() - start) // 1_000_000

        result = r.json()
        content = ""
        if result.get("output"):
            for item in result["output"]:
                if item.get("content"):
                    for c in item["content"]:
                        if c.get("text"):
                            content += c["text"]

        usage = result.get("usage", {})
        return TaskResult(
            task_id=task.id,
            model=self.model_name,
            response=content,
            score=0.0,
            latency_ms=latency_ms,
            tokens_in=usage.get("input_tokens", 0),
            tokens_out=usage.get("output_tokens", 0),
        )


class BedrockConverseRunner(ModelRunner):
    """Runner for Bedrock via the provider-agnostic Converse API.

    Works for inference-profile-only models (OpenAI GPT-5.6 Sol/Terra/Luna,
    Claude Opus 5, Nova 2, etc.) that reject on-demand invoke_model. Reasoning
    models return a reasoningContent block before the text block; we skip it.
    """

    def __init__(self, model_id: str = "global.openai.gpt-5.6-sol", region: str = "eu-central-1"):
        import boto3
        from botocore.config import Config
        # Optional "model@region" suffix overrides the default region (some models are region-scoped)
        if "@" in model_id:
            model_id, region = model_id.rsplit("@", 1)
        self._model_id = model_id
        self._region = region
        # Reasoning models can take >60s per turn; extend read timeout and retry throttles
        cfg = Config(read_timeout=300, connect_timeout=15, retries={"max_attempts": 4, "mode": "adaptive"})
        self._client = boto3.client("bedrock-runtime", region_name=region, config=cfg)

    @property
    def model_name(self) -> str:
        # Strip routing prefix (global./us./eu.) and provider namespace for display
        short = self._model_id
        for prefix in ("global.", "us.", "eu.", "apac."):
            if short.startswith(prefix):
                short = short[len(prefix):]
                break
        if "." in short:
            short = short.split(".", 1)[1]
        return f"bedrock/{short}"

    def run(self, task: Task) -> TaskResult:
        start = time.perf_counter_ns()
        response = self._client.converse(
            modelId=self._model_id,
            system=[{"text": SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": task.prompt}]}],
            inferenceConfig={"maxTokens": 4096},
        )
        latency_ms = (time.perf_counter_ns() - start) // 1_000_000

        # Extract only text blocks; reasoning models emit a reasoningContent block first
        content = ""
        for block in response.get("output", {}).get("message", {}).get("content", []):
            if "text" in block:
                content += block["text"]

        usage = response.get("usage", {})
        return TaskResult(
            task_id=task.id,
            model=self.model_name,
            response=content,
            score=0.0,
            latency_ms=latency_ms,
            tokens_in=usage.get("inputTokens", 0),
            tokens_out=usage.get("outputTokens", 0),
        )


class OllamaCompletionRunner(ModelRunner):
    """Runner for Ollama base models (completion mode, no chat)."""

    def __init__(self, model: str = "llama3-german-8b", base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = base_url

    @property
    def model_name(self) -> str:
        return f"ollama-completion/{self._model}"

    def run(self, task: Task) -> TaskResult:
        prompt = f"Frage: {task.prompt}\n\nAntwort:"
        start = time.perf_counter_ns()
        resp = httpx.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 1024},
            },
            timeout=300.0,
        )
        latency_ms = (time.perf_counter_ns() - start) // 1_000_000
        data = resp.json()

        return TaskResult(
            task_id=task.id,
            model=self.model_name,
            response=data.get("response", ""),
            score=0.0,
            latency_ms=latency_ms,
            tokens_in=data.get("prompt_eval_count", 0),
            tokens_out=data.get("eval_count", 0),
        )


def get_runner(model_spec: str) -> ModelRunner:
    """Parse 'provider/model' string and return appropriate runner."""
    if "/" not in model_spec:
        raise ValueError(f"Model spec must be 'provider/model', got: {model_spec}")

    provider, model = model_spec.split("/", 1)

    match provider:
        case "openai":
            return OpenAIRunner(model=model)
        case "ollama":
            return OllamaRunner(model=model)
        case "ollama-completion":
            return OllamaCompletionRunner(model=model)
        case "bedrock":
            if "nova" in model:
                return BedrockNovaRunner(model_id=model)
            if "mistral" in model or "mixtral" in model or "ministral" in model or "pixtral" in model:
                return BedrockMistralRunner(model_id=model)
            return BedrockRunner(model_id=model)
        case "bedrock-mantle":
            return BedrockMantleRunner(model_id=model)
        case "bedrock-converse":
            return BedrockConverseRunner(model_id=model)
        case _:
            raise ValueError(f"Unknown provider: {provider}. Supported: openai, ollama, ollama-completion, bedrock, bedrock-converse, bedrock-mantle")
