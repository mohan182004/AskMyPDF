from langchain_core.language_models.llms import LLM
from typing import Optional, List, Any
from openai import OpenAI

class GithubGPT41LLM(LLM):
    api_key: str
    api_url: str
    model: str = "openai/gpt-4.1"
    temperature: float = 1.0
    top_p: float = 1.0

    @property
    def _llm_type(self) -> str:
        return "github-gpt-41"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        client = OpenAI(
            base_url=self.api_url,
            api_key=self.api_key,
        )
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            model=self.model
        )
        return response.choices[0].message.content