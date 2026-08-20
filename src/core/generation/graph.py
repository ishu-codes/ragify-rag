"""RAG generation graph.

Nodes are plain state functions (``state -> partial updates``), exactly like
LangGraph expects: the framework owns the shared state and merges whatever each
node returns.  The :class:`RagGraph` class is only a factory: it holds the
node registry, the dependencies (LLM, retriever factory, tools, ...) and the
edge wiring, and compiles the graph lazily.  It is intentionally stateless
across invocations so that concurrent calls (e.g. the gRPC server's worker
pool) can share one compiled graph safely.

Node methods are registered with the :func:`node` decorator.  The decorator
wraps each method with :func:`guarded` for uniform error reporting and stores
the wrapped method in the node registry, mirroring the original
``graph.add_node(name, _guarded(node))`` style.  Edges are declared on the same
decorator: ``end_key`` sets the target node (or a routing callable for a
conditional edge), and ``is_start=True`` marks the entry point:

    @node("query_analysis")
    def query_classifier(self, state: State) -> dict[str, Any]:
        ...
"""

import re
import threading
import types
from functools import wraps
from os import getenv
from typing import Any, Callable

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph
from tavily import TavilyClient

from src.core.classification.model import classification_model
from src.core.generation.events import end_event, end_pipeline, error_event, start_event
from src.core.generation.llm import llm
from src.core.generation.prompts import prompts
from src.core.generation.schema import Evaluate
from src.core.generation.state import State
from src.core.generation.tools import doc_tool, routing_tool
from src.core.retrieval import get_retriever
from src.core.utils.logger import get_logger
from src.core.utils.text_quality import is_degenerate

logger = get_logger("ragify.graph")

Node = Callable[[State], dict[str, Any]]

# Node name -> guarded node method on RagGraph, populated by node().
_NODES: dict[str, Node] = {}
# Edges: {source: target}. Source is a node name or START; target is a node
# name, END, or a routing callable (which produces a conditional edge).
_EDGES: dict[Any, Any] = {}


def node(
    name: str | None = None, end_key: str | None = None, is_start: bool = False
) -> Callable[[Node], Node]:
    """Register a node method so it is added to the graph at build time.

    The node name defaults to the method name when not given explicitly.
    Registering the same name twice with different methods fails fast.

    Edge wiring is declared here too:
    - ``end_key`` is the target node name, END, or a routing callable for a
      conditional edge (defaults to END when omitted);
    - ``is_start=True`` marks this node as the graph's entry point.
    """

    def decorator(method: Node) -> Node:
        node_name = name or method.__name__
        if node_name in _NODES:
            raise ValueError(f"Node {node_name!r} is already registered")
        _NODES[node_name] = guarded(method)

        if is_start:
            if START in _EDGES and _EDGES[START] != node_name:
                raise ValueError(
                    f"Graph already starts at {_EDGES[START]!r}; "
                    f"cannot also start at {node_name!r}"
                )
            _EDGES[START] = node_name
        _EDGES[node_name] = end_key or END
        return method

    return decorator


def guarded(method: Node) -> Node:
    """Wrap a node method to log failures via error_event and re-raise.

    Failures are never swallowed: LangGraph propagates the exception so the
    caller (e.g. the gRPC server) can report a proper error to the client.
    """

    @wraps(method)
    def wrapper(self, state: State) -> dict[str, Any]:
        try:
            return method(self, state)
        except Exception as exc:
            error_event(str(exc), method.__name__)
            raise

    return wrapper


def _trim_context(text: str, limit: int = 2000) -> str:
    text = text.strip()
    max_end_start_time = max(limit, len(text) - limit)
    if len(text) > limit:
        return (
            text[:limit] + "\n\n...[context truncated]\n\n" + text[max_end_start_time:]
        )
    return text


def _question(state: State) -> str:
    return state.get("messages", [{}])[-1].content


class RagGraph:
    """Factory for the RAG state graph.

    The class owns construction (dependencies and edge wiring) and exposes a
    single thread-safe :meth:`build` entry point.  All dependencies are
    injected so tests can substitute fakes for the LLM, retriever and tools.
    """

    def __init__(
        self,
        *,
        llm: Any,
        tavily: TavilyClient,
        prompts: Any,
        classifier: Any,
        get_retriever: Callable[[Any], Any],
    ) -> None:
        self._llm = llm
        self._tavily = tavily
        self._prompts = prompts
        self._classifier = classifier
        self._get_retriever = get_retriever
        self._build_lock = threading.Lock()
        self._compiled: Any = None

    # ----- Nodes -----------------------------------------------------------

    # Query Analysis

    @node("query_analysis", end_key=routing_tool, is_start=True)
    def query_classifier(self, state: State) -> dict[str, Any]:
        workspace_id = state.get("workspace_id")
        question = _question(state)
        start_event(
            "Classifying user query",
            f"workspace_id: {workspace_id}\nquery: {question[:500]}",
        )

        retriever = self._get_retriever(workspace_id)
        context = retriever.invoke({"query": question})
        # The langchain retriever tool returns page contents joined by "\n\n"
        # (a string, not a list of Documents). Split it back into chunks so
        # the dedupe loop below sees real text instead of single characters.
        if isinstance(context, str):
            context = [chunk for chunk in context.split("\n\n") if chunk]

        # Deduplicate retrieved chunks and cap the prompt size so the LLM call
        # stays fast.
        seen = set()
        unique_chunks = []
        for doc in context:
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            if content not in seen:
                seen.add(content)
                unique_chunks.append(content)
        raw_context = "\n\n".join(unique_chunks)

        # Fast path: questions that clearly reference the retrieved context or
        # an uploaded document should use it without spending a classifier LLM
        # call.  This is intentionally generic (any document type), so
        # domain-specific queries are left to the classifier.
        references_document = re.search(
            r"\bthis\s+(?:document|file|text|content|passage|context)\b|"
            r"the\s+(?:given|uploaded|attached|provided|above)\s+(?:document|file|text|content|passage)\b|"
            r"according to (?:the\s+)?(?:context|document|file)|"
            r"based on (?:the\s+)?(?:context|document)|"
            r"refer(?:ring|ence)? to (?:the\s+)?(?:document|file|context)|"
            r"\bfrom\s+the\s+(?:document|file|text|content|passage|context|pdf)\b|"
            r"\b(?:in|within)\s+the\s+(?:document|file|text|content|passage|context|pdf)\b|"
            r"\bwhat\s+does\s+the\s+(?:document|file|pdf)\s+say\b|"
            r"\bthe\s+(?:document|file|pdf)\s+(?:says|states|mentions|describes|explains|indicates)\b",
            question,
            re.IGNORECASE,
        )
        if references_document:
            # Document-referencing queries always take the document path, even
            # when the retrieved context looks degenerate: the generator's
            # quality gate will answer honestly that nothing relevant was
            # found instead of switching to a context-free general answer.
            if is_degenerate(raw_context):
                logger.warning(
                    "degenerate_retrieved_context",
                    extra={
                        "workspace_id": workspace_id,
                        "context_chars": len(raw_context),
                        "chunk_count": len(unique_chunks),
                        "query_references_document": True,
                    },
                )
                end_event("Route: index (document-referencing question, degenerate context)")
            else:
                end_event("Route: index (document-referencing question)")
            return {
                "route": "index",
                "latest_query": question,
                "context": [raw_context],
            }

        # Quality gate: if extraction garbage made it into the index, do not
        # classify or answer from it. Degrade to the general path explicitly.
        if is_degenerate(raw_context):
            logger.warning(
                "degenerate_retrieved_context",
                extra={
                    "workspace_id": workspace_id,
                    "context_chars": len(raw_context),
                    "chunk_count": len(unique_chunks),
                },
            )
            end_event("Route: general (retrieved context is degenerate)")
            return {
                "route": "general",
                "latest_query": question,
                "context": [],
            }

        context = _trim_context(raw_context)
        logger.info(
            "docs_retrieved",
            extra={
                "workspace_id": workspace_id,
                "chunk_count": len(unique_chunks),
                "preview": context[:120],
            },
        )

        classify_prompt = PromptTemplate(
            template=self._prompts.classify_prompt,
            input_variables=["question", "context"],
        )
        classifier_chain = classify_prompt | self._classifier
        fallback_chain = classify_prompt | self._llm.client

        try:
            result = classifier_chain.invoke(
                {"question": question, "context": context}
            ).content
        except Exception as classifier_exc:
            # The dedicated classifier endpoint may be degraded (e.g. NVIDIA
            # returns 400 "DEGRADED function cannot be invoked"). Retry with
            # the main LLM so routing is not silently reduced to a fallback.
            logger.warning(
                "classifier_unavailable",
                extra={
                    "error": str(classifier_exc)[:200],
                },
            )
            try:
                result = fallback_chain.invoke(
                    {"question": question, "context": context}
                ).content
            except Exception as exc:
                error_event(str(exc), "query classifier")

                # If both classifiers fail, prefer using the retrieved context
                # over falling back to a context-free answer.
                route = "index" if context else "general"
                end_event(f"Route: {route} (classifier fallback)")
                return {
                    "route": route,
                    "latest_query": question,
                    "context": [context],
                }

        match = re.search(r"\s*['\"]?(index|general|search)['\"]?", str(result))
        route = match.group(1) if match else "index"

        end_event(f"Route: {route}\nclassifier output: {result[:500]}")
        return {
            "route": route,
            "latest_query": question,
            "context": [context],
        }

    # General LLM

    @node("general_llm", end_key=END)
    def general_llm(self, state: State) -> dict[str, Any]:
        question = _question(state)
        start_event("Generating response (general)", f"query: {question[:500]}")

        result = self._llm.invoke(state["messages"])
        end_event(result.content)
        end_pipeline()
        return {"messages": [result]}

    # Retriever

    @node("retriever", end_key="evaluator")
    def retriever_node(self, state: State) -> dict[str, Any]:
        messages = state.get("latest_query", "")
        workspace_id = state.get("workspace_id")

        # Imported lazily to keep module import light and avoid import cycles.
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

    # Evaluator

    @node("evaluator", end_key=doc_tool)
    def evaluator(self, state: State) -> dict[str, Any]:
        context = state.get("context", [])
        context = _trim_context("\n\n".join(str(part) for part in context))

        question = state.get("latest_query", "")
        start_event(
            "Evaluating response",
            f"question: {question[:500]}\ncontext: {_trim_context(context, 600)}",
        )

        grading_prompt = PromptTemplate(
            template=self._prompts.grading_prompt,
            input_variables=["question", "context"],
        )

        llm_with_grade = self._llm.with_structured_output(Evaluate)
        chain_graded = grading_prompt | llm_with_grade
        result = chain_graded.invoke({"question": question, "context": str(context)})

        end_event(f"Retrieval evaluator: {result}")
        return {"binary_score": result["binary_score"]}

    # Refinement

    @node("refinement", end_key="retriever")
    def query_refinement(self, state: State) -> dict[str, Any]:
        query = state.get("latest_query", "")
        start_event("Refining response", f"query: {query[:500]}")

        rewrite_prompt = PromptTemplate(
            template=self._prompts.rewrite_prompt, input_variables=["query"]
        )
        chain = rewrite_prompt | self._llm.client
        result = chain.invoke({"query": query})
        end_event(f"Refined query: {result.content[:500]}")

        return {
            "latest_query": result.content,
            "refinement_count": (state.get("refinement_count") or 0) + 1,
        }

    # Web search

    @node("web_search", end_key="generator")
    def web_search(self, state: State) -> dict[str, Any]:
        query = state.get("latest_query") or ""
        start_event("Searching the web", f"query: {query[:500]}")

        results = self._tavily.search(query, timeout=30).get("results", [])

        first = f"\nfirst result: {results[0].get('title')}" if results else ""
        end_event(f"Web search: {len(results)} result(s){first}")

        websearch_result = "web search results:\n" + "\n\n".join(
            [
                f"{result.get('title')} ({result.get('url')})\n{result.get('content')}"
                for result in results
            ]
        )

        return {"messages": [AIMessage(content=websearch_result)]}

    # Generator

    @node("generator", end_key=END)
    def generate(self, state: State) -> dict[str, Any]:
        retrieved = state.get("context", [])
        retrieved_text = "\n\n".join(str(part) for part in retrieved)
        if is_degenerate(retrieved_text):
            logger.warning(
                "degenerate_generation_context",
                extra={"workspace_id": state.get("workspace_id")},
            )
            answer = "I couldn't find relevant content in the documents to answer this question."
            end_event(answer)
            end_pipeline()
            return {"messages": [AIMessage(content=answer)]}

        context = retrieved
        messages = state.get("messages", [{}])
        # Handle both dict and AIMessage/Message objects
        message_contents = []
        for msg in messages:
            if hasattr(msg, "content"):
                message_contents.append(msg.content)
            elif isinstance(msg, dict):
                message_contents.append(msg.get("content", ""))
            else:
                message_contents.append(str(msg))
        message_contents.extend(context)

        context = "\n\n\n".join(message_contents)
        start_event("Generating response", f"context: {_trim_context(context, 1000)}")

        generate_prompt = PromptTemplate(
            template=self._prompts.generate_prompt, input_variables=["context"]
        )
        generate_chain = generate_prompt | self._llm.client
        result = generate_chain.invoke({"context": context})

        end_event(result.content)
        end_pipeline()
        return {"messages": [result]}

    # ----- Construction ----------------------------------------------------

    def build(self) -> Any:
        """Compile the graph once and return it.

        The compiled graph is immutable and safe to share across threads, so
        this is called lazily and cached (double-checked locking keeps it safe
        even if two threads race on the first build).
        """
        if self._compiled is not None:
            return self._compiled

        with self._build_lock:
            if self._compiled is not None:
                return self._compiled

            graph = StateGraph(State)

            # Register nodes (bind the stored guarded method to this instance)
            for node_name, node_fn in _NODES.items():
                graph.add_node(node_name, types.MethodType(node_fn, self))

            # Register edges
            for source, target in _EDGES.items():
                if callable(target):
                    graph.add_conditional_edges(source, target)
                else:
                    graph.add_edge(source, target)

            self._compiled = graph.compile()

        return self._compiled


# Backwards-compatible facade: src.core.generation and the gRPC server import
# `builder` directly, so keep the same module-level name.
builder = RagGraph(
    llm=llm,
    tavily=TavilyClient(api_key=getenv("TAVILY_API_KEY")),
    prompts=prompts,
    classifier=classification_model.client,
    get_retriever=get_retriever,
).build()
