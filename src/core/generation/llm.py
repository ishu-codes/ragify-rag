from os import environ

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from src.core.utils.config import (
    LLM_API_KEY,
    LLM_MODEL,
    LLM_STRUCTURED_OUTPUT,
    LLM_URL,
)

load_dotenv()

environ["OPENAI_API_KEY"] = LLM_API_KEY

class LLM:
    def __init__(
        self,
    ):
        self.model_name = LLM_MODEL
        self._client = ChatOpenAI(
            model=self.model_name,
            base_url=LLM_URL,
            api_key=LLM_API_KEY,
            timeout=60,
            max_retries=2,
        )

    @property
    def client(self):
        return self._client

    def invoke(self, *args, **kwargs):
        return self._client.invoke(*args, **kwargs)

    def bind_tools(self, tools):
        return self._client.bind_tools(tools)

    def with_structured_output(self, schema):
        if LLM_STRUCTURED_OUTPUT:
            return self._client.with_structured_output(schema)

        # Provider-agnostic fallback: parse JSON from the model output
        # without sending a response_format that some APIs reject.
        return self._client | JsonOutputParser(pydantic_object=schema)

    def __getattr__(self, name):
        return getattr(self._client, name)


llm = LLM()
