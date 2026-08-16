from pydantic import BaseModel, Field


class Evaluate(BaseModel):
    binary_score: str = Field(description="Relevance score: 'yes' or 'no'")


class RouteIdentifier(BaseModel):
    route: str = Field(description="Route: 'index', 'general', or 'web'")


class VerificationResult(BaseModel):
    faithful: bool = Field(description="True if answer is supported by the context.")
    explanation: str = Field(description="Brief reasoning.")
