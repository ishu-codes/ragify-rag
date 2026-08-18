import re
from os import getenv

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import PromptTemplate

# from langchain_tavily import TavilySearch
from langgraph.graph import END, START
from langgraph.graph.state import StateGraph
from tavily import TavilyClient

from src.core.classification.model import classification_model
from src.core.generation.events import end_event, end_pipeline, error_event, start_event
from src.core.generation.llm import llm
from src.core.generation.prompts import prompts
from src.core.generation.schema import Evaluate, RouteIdentifier
from src.core.generation.state import State
from src.core.generation.tools import doc_tool, routing_tool
from src.core.retrieval import get_retriever

tavily_client = TavilyClient(api_key=getenv("TAVILY_API_KEY"))


def _trim_context(text: str, limit: int = 4000) -> str:
    text = text.strip()
    if len(text) > limit:
        return text[:limit] + "\n...[context truncated]"
    return text


def _question(state: State) -> str:
    return state.get("messages", [{}])[-1].content


def query_classifier(state: State):
    workspace_id = state.get("workspace_id")
    question = _question(state)
    start_event(
        "Classifying user query",
        f"workspace_id: {workspace_id}\nquery: {question[:500]}",
    )

    retriever = get_retriever(workspace_id)
    context = retriever.invoke({"query": question})

    # Deduplicate retrieved chunks and cap the prompt size so the LLM call stays fast.
    seen = set()
    unique_chunks = []
    for doc in context:
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        if content not in seen:
            seen.add(content)
            unique_chunks.append(content)
    context = _trim_context("\n\n".join(unique_chunks))

    # Fast path: questions that clearly reference the retrieved context or an
    # uploaded document should use it without spending a classifier LLM call.
    # This is intentionally generic (any document type), so domain-specific
    # queries are left to the classifier.
    references_document = re.search(
        r"\bthis\s+(?:document|file|text|content|passage|context)\b|"
        r"the\s+(?:given|uploaded|attached|provided|above)\s+(?:document|file|text|content|passage)\b|"
        r"according to (?:the\s+)?(?:context|document|file)|"
        r"based on (?:the\s+)?(?:context|document)|"
        r"refer(?:ring|ence)? to (?:the\s+)?(?:document|file|context)",
        question,
        re.IGNORECASE,
    )
    if context and references_document:
        end_event("Route: index (document-referencing question)")
        return {
            "messages": state["messages"],
            "route": "index",
            "latest_query": question,
            "context": [context],
        }

    classify_prompt = PromptTemplate(
        template=prompts.classify_prompt,
        input_variables=["question", "context"],
    )
    # classify = classification_model.classify(RouteIdentifier)
    classify = classification_model.client
    chain = classify_prompt | classify

    try:
        result = chain.invoke({"question": question, "context": context}).content

        match = re.search(r"\s*['\"]?(index|general|search)['\"]?", str(result))
        route = match.group(1) if match else 'index'

        end_event(f"Route: {route}\nclassifier output: {result[:500]}")
        return {
            "messages": state["messages"],
            "route": route,
            "latest_query": question,
            "context": [context],
        }

    except Exception as e:
        error_event(str(e), "query classifier")

        # If the classifier fails (e.g. LLM timeout), prefer using the retrieved
        # context over falling back to a context-free general answer.
        route = "index" if context else "general"
        end_event(f"Route: {route} (classifier fallback)")
        return {
            "messages": state["messages"],
            "route": route,
            "latest_query": question,
            "context": [context],
        }


def general_llm(state: State):
    question = _question(state)
    start_event("Generating response (general)", f"query: {question[:500]}")

    result = llm.invoke(state["messages"])
    end_event(result.content)
    end_pipeline()
    return {"messages": [result]}


def retriever_node(state: State):
    messages = state.get("latest_query", "")
    workspace_id = state.get("workspace_id")
    from src.core.generation.agent import get_agent

    start_event(
        "Retrieving context from vector db",
        f"workspace_id: {workspace_id}\nquery: {messages[:500]}",
    )

    agent = get_agent(workspace_id)
    result = agent.invoke({"messages": [{"role": "user", "content": messages}]})

    output = result.get("messages", "")[-1]
    if isinstance(output, BaseMessage):
        output = output.content

    tool_calls = []
    tool_results = []
    for message in result.get("messages", []):
        for call in getattr(message, "tool_calls", []) or []:
            tool_calls.append({"tool": call["name"], "input": call.get("args", {})})
        if isinstance(message, ToolMessage):
            tool_results.append(message.content)

    new_message = AIMessage(
        content=output, additional_kwargs={"tool_calls": tool_calls}
    )
    end_event(
        f"Retrieved {len(tool_results)} chunk(s)\n"
        f"result: {_trim_context(str(output), 1000)}"
    )

    return {"messages": [new_message], "context": [output, *tool_results]}


def evaluator(state: State):
    context = state.get('context', [])
    context = _trim_context("\n\n".join(str(part) for part in context))

    question = state.get("latest_query", "")
    start_event(
        "Evaluating response",
        f"question: {question[:500]}\ncontext: {_trim_context(context, 600)}",
    )

    grading_prompt = PromptTemplate(
        template=prompts.grading_prompt,
        input_variables=["question", "context"],
    )

    llm_with_grade = llm.with_structured_output(Evaluate)
    chain_graded = grading_prompt | llm_with_grade
    result = chain_graded.invoke({"question": question, "context": str(context)})

    end_event(f"Retrieval evaluator: {result}")
    return {"messages": state["messages"], "binary_score": result["binary_score"]}


def query_refinement(state: State):
    query = state.get("latest_query", "")
    start_event("Refining response", f"query: {query[:500]}")

    rewrite_prompt = PromptTemplate(
        template=prompts.rewrite_prompt, input_variables=["query"]
    )
    chain = rewrite_prompt | llm.client
    result = chain.invoke({"query": query})
    end_event(f"Refined query: {result.content[:500]}")

    return {
        "latest_query": result.content,
        "refinement_count": (state.get("refinement_count") or 0) + 1,
    }


def web_search(state: State):
    query = state.get("latest_query", "")
    start_event("Searching the web", f"query: {query[:500]}")

    results = tavily_client.search(query, timeout=30).get("results", [])

    first = f"\nfirst result: {results[0].get('title')}" if results else ""
    end_event(f"Web search: {len(results)} result(s){first}")

    websearch_result = "web search results:\n" + "\n\n".join([
        f"{result.get('title')} ({result.get('url')})\n{result.get('content')}"
        for result in results
    ])

    return {"messages": [AIMessage(content=websearch_result)]}


def generate(state: State):
    context = state.get('context', [])
    messages = state.get("messages", [{}])
    # Handle both dict and AIMessage/Message objects
    message_contents = []
    for msg in messages:
        if hasattr(msg, 'content'):
            message_contents.append(msg.content)
        elif isinstance(msg, dict):
            message_contents.append(msg.get('content', ''))
        else:
            message_contents.append(str(msg))
    message_contents.extend(context)

    context = "\n\n\n".join(message_contents)
    start_event("Generating response", f"context: {_trim_context(context, 1000)}")

    generate_prompt = PromptTemplate(
        template=prompts.generate_prompt,
        input_variables=["context"]
    )
    generate_chain = generate_prompt | llm.client
    result = generate_chain.invoke({"context": context})

    end_event(result.content)
    end_pipeline()
    return {"messages": [result]}


graph = StateGraph(State)


def _guarded(node):
    """Run a graph node and print the error detail if it fails."""

    def wrapper(state):
        try:
            return node(state)
        except Exception as exc:
            error_event(str(exc), node.__name__)
            raise

    return wrapper


graph.add_node("query_analysis", _guarded(query_classifier))
graph.add_node("retriever", _guarded(retriever_node))
graph.add_node("evaluator", _guarded(evaluator))
graph.add_node("generator", _guarded(generate))
graph.add_node("refinement", _guarded(query_refinement))
graph.add_node("web_search", _guarded(web_search))
graph.add_node("general_llm", _guarded(general_llm))

graph.add_edge(START, "query_analysis")
graph.add_conditional_edges("query_analysis", routing_tool)
graph.add_edge("general_llm", END)

graph.add_edge("retriever", "evaluator")
graph.add_conditional_edges("evaluator", doc_tool)
graph.add_edge("refinement", "retriever")

graph.add_edge("web_search", "generator")
graph.add_edge("generator", END)


builder = graph.compile()
