"""Inference through any OpenAI-compatible API (OpenAI, Groq, OpenRouter, DeepSeek)."""

from typing import Any

import httpx

from trustnet.providers.base import (
    GenerationRequest,
    GenerationResponse,
    ProviderConfigurationError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


class OpenAIProvider:
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 60.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ProviderConfigurationError("an API key is required")
        if not model or not model.strip():
            raise ProviderConfigurationError("a model name is required")
        if timeout_seconds <= 0:
            raise ProviderConfigurationError("the timeout must be greater than zero")
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds
        self._client = http_client

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature,
            "response_format": {"type": "json_object"},
        }
        if request.seed is not None:
            payload["seed"] = request.seed

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/chat/completions"
        client = self._client or httpx.Client(timeout=self.timeout)
        try:
            resp = client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"API model '{self.model}' did not finish replying within {self.timeout:g} seconds."
            ) from exc
        except (httpx.ConnectError, ConnectionError) as exc:
            raise ProviderUnavailableError(
                f"Cannot reach API at {self.base_url}: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"HTTP request to {self.base_url} failed: {exc}") from exc
        finally:
            if self._client is None:
                client.close()

        if resp.status_code == 401:
            raise ProviderConfigurationError(
                f"Authentication failed for {self.base_url}. Check that your API key is valid."
            )
        if resp.status_code != 200:
            error_msg = resp.text
            try:
                err_json = resp.json()
                if "error" in err_json and "message" in err_json["error"]:
                    error_msg = err_json["error"]["message"]
            except Exception:
                pass
            raise ProviderError(f"API request failed (HTTP {resp.status_code}): {error_msg}")

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise ProviderError(f"API model '{self.model}' returned no choices in response.")

        reply_content = choices[0].get("message", {}).get("content", "")
        if not reply_content or not reply_content.strip():
            raise ProviderError(f"API model '{self.model}' returned an empty reply.")

        return GenerationResponse(text=reply_content, provider=self.name, model=self.model)
