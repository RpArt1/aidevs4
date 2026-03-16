from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

from common.logger import get_logger

log = get_logger(__name__)

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    def __iadd__(self, other: "TokenUsage") -> "TokenUsage":
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        return self

    def __str__(self) -> str:
        return (
            f"in: {self.input_tokens}  out: {self.output_tokens}  "
        )


class LLMService:
    def __init__(self, model: str = "openai/gpt-4o"):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in environment")

        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
        )
        self.last_usage = TokenUsage()
        self.total_usage = TokenUsage()
        log.info(f"Initialized LLMService with model: {self.model}")

    def _track_usage(self, response) -> None:
        usage = response.usage
        if usage:
            self.last_usage = TokenUsage(
                input_tokens=usage.prompt_tokens or 0,
                output_tokens=usage.completion_tokens or 0,
            )
        else:
            self.last_usage = TokenUsage()
        self.total_usage += self.last_usage


    def chat(self, messages: list[dict], **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        self._track_usage(response)
        log.info(f"[LLMService] model={self.model} | last_usage=({self.last_usage}) | total_usage=({self.total_usage})")
        return response.choices[0].message.content

    def chat_structured(
        self,
        messages: list[dict],
        schema: dict,
        system_prompt: str | None = None,
        **kwargs,
    ) -> dict:
        """Call LLM with a JSON schema for structured output.

        Args:
            schema: A dict with "name", "schema", and optionally "strict" keys.
                    Example: {"name": "classification", "strict": True, "schema": {...}}
            system_prompt: Optional system message prepended to the conversation.
        """
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}, *messages]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": schema,
            },
            **kwargs,
        )
        self._track_usage(response)
        log.info(f"[LLMService] model={self.model} | last_usage=({self.last_usage}) | total_usage=({self.total_usage})")
        return json.loads(response.choices[0].message.content)
