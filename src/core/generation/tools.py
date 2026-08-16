from typing import Literal

from langchain_core.prompts import PromptTemplate

from src.ragify.generation.llm import llm
from src.ragify.generation.prompts import prompts
from src.ragify.generation.schema import Evaluate, RouteIdentifier, VerificationResult

MAX_REFINEMENTS = 1


def route_query(state: dict) -> Literal["evaluator", "general_llm", "web_search"]:
    route = state.get("route", "general")
    if route == "index":
        # return "retriever"
        return "evaluator"
    elif route == "general":
        return "general_llm"
    return "web_search"


def grade_documents(state: dict) -> Literal["refinement", "generator"]:
    score = state.get("binary_score", "no")
    if score == "yes":
        return "generator"

    refinements = state.get("refinement_count", 0) or 0
    if refinements >= MAX_REFINEMENTS:
        print(f"[doc_tool] Max refinements reached ({refinements}), routing to generator")
        return "generator"

    print(f"[doc_tool] Routing based on score: {score} (refinement {refinements})")
    return "refinement"


def verify_answer(state: dict) -> Literal["__end__", "generate"]:
    if state.get("route") == "general":
        return "__end__"

    question = state.get("latest_query", "")
    context = state.get("messages", [{}])[-1].get("content", "")
    final_answer = context

    verify_prompt = PromptTemplate(
        template=prompts.verify_prompt,
        input_variables=["question", "context", "final_answer"],
    )
    llm_with_verification = llm.with_structured_output(VerificationResult)
    verify_chain = verify_prompt | llm_with_verification

    result = verify_chain.invoke(
        {"question": question, "context": context, "final_answer": final_answer}
    )

    if result["faithful"]:
        return "__end__"
    print("Generating again as answer is not faithful.")
    return "generate"


routing_tool = route_query
doc_tool = grade_documents
