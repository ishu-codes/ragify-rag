
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from src.ragify.utils.config import (
    CLASSIFICATION_API_KEY,
    CLASSIFICATION_MODEL,
    CLASSIFICATION_URL,
)
from src.ragify.generation.schema import RouteIdentifier


class ClassificationModel:
    def __init__(self) -> None:
        self._model_name = CLASSIFICATION_MODEL
        self._client = ChatOpenAI(
            model=CLASSIFICATION_MODEL,
            base_url=CLASSIFICATION_URL,
            api_key=CLASSIFICATION_API_KEY,
            timeout=30,
            max_retries=2,
        )

    @property
    def client(self):
        return self._client

    def classify(self, schema):
        # Use JSON mode without response_format for provider compatibility.
        prompt = PromptTemplate(
            template="Classify the query. Available routes: index, general, web. Query: {question} Context: {context}",
            input_variables=["question", "context"],
        )
        parser = JsonOutputParser(pydantic_object=schema)
        return prompt | self._client | parser


classification_model = ClassificationModel()
