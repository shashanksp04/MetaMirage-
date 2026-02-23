from openai import OpenAI
import re

class RClient:
    def __init__(self,
                 model_name="Qwen3-32B",
                 openai_api_key="token-abc123",
                 openai_api_base="None",
                 messages=None):
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.model_name = model_name
        self.messages = messages or []
        self.history = []

    def chat(self, prompt, temperature=0.6, max_tokens=10000):
        self.messages.append({"role": "user", "content": prompt})
        self.history.append({"role": "user", "content": prompt})

        create_kwargs = {
            "model": self.model_name,
            "messages": self.messages,
        }
        common_args = {
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }

        resp = self.client.chat.completions.create(
            **create_kwargs,
            **common_args
        )
        content = resp.choices[0].message.content
        reasoning = resp.choices[0].message.reasoning_content

        self.messages.append({"role": "assistant", "content": content})
        self.history.append({"role": "assistant", "content": content})

        return {"content": content, "reasoning": reasoning}

    def get_history(self):
        return self.history

    def info(self):
        return {
            "model_name": self.model_name,
        }