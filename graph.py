"""LangGraph agentic RAG workflow (document-only).

START -> route -+-> off_topic -> END
                |
                +-> retrieve -> grade_documents -+-> generate -> END
                       ^                         |
                       |                         +-> (no relevant docs, retries left)
                       +------ rewrite_query <---+
                                                 +-> not_found -> END (retries exhausted)
"""
from typing import List, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

import config
import prompts

OFF_TOPIC_MESSAGE = (
    "This question isn't related to the uploaded document, so I can't answer it. "
    "Please ask something about the document."
)


class RouteDecision(BaseModel):
    datasource: Literal["document", "off_topic"] = Field(description="Where to send the question")


class Grade(BaseModel):
    relevant: Literal["yes", "no"] = Field(description="Is the passage relevant?")


class GraphState(TypedDict, total=False):
    question: str
    query: str
    documents: List[Document]
    generation: str
    rewrites: int


def _document_preview(vectorstore, n: int = 3, max_chars: int = 400) -> str:
    """A few chunks from the document, so the router knows what it is about."""
    try:
        chunks = list(vectorstore.docstore._dict.values())[:n]
        return "\n...\n".join(c.page_content[:max_chars] for c in chunks)
    except Exception:
        return "(preview unavailable)"


def build_graph(vectorstore, api_key: str):
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL, google_api_key=api_key, temperature=0
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})
    preview = _document_preview(vectorstore)

    router = ChatPromptTemplate.from_template(prompts.ROUTER_PROMPT) | llm.with_structured_output(RouteDecision)
    grader = ChatPromptTemplate.from_template(prompts.GRADER_PROMPT) | llm.with_structured_output(Grade)
    rewriter = ChatPromptTemplate.from_template(prompts.REWRITE_PROMPT) | llm | StrOutputParser()
    generator = ChatPromptTemplate.from_template(prompts.GENERATE_PROMPT) | llm | StrOutputParser()

    # ---- nodes ----
    def route_node(state: GraphState):
        return {"query": state["question"], "rewrites": 0, "documents": []}

    def retrieve(state: GraphState):
        return {"documents": retriever.invoke(state["query"])}

    def grade_documents(state: GraphState):
        inputs = [
            {"question": state["question"], "document": d.page_content}
            for d in state["documents"]
        ]
        grades = grader.batch(inputs)
        kept = [d for d, g in zip(state["documents"], grades) if g.relevant == "yes"]
        return {"documents": kept}

    def rewrite_query(state: GraphState):
        new_query = rewriter.invoke({"question": state["question"]})
        return {"query": new_query.strip(), "rewrites": state.get("rewrites", 0) + 1}

    def generate(state: GraphState):
        context = "\n\n".join(d.page_content for d in state["documents"])
        return {"generation": generator.invoke({"context": context, "question": state["question"]})}

    def off_topic(state: GraphState):
        return {"generation": OFF_TOPIC_MESSAGE, "documents": []}

    def not_found(state: GraphState):
        return {"generation": "I couldn't find relevant information about that in the document.", "documents": []}

    # ---- conditional edges ----
    def decide_route(state: GraphState) -> Literal["retrieve", "off_topic"]:
        decision = router.invoke({"question": state["question"], "preview": preview})
        return "retrieve" if decision.datasource == "document" else "off_topic"

    def decide_after_grading(state: GraphState) -> Literal["generate", "rewrite_query", "not_found"]:
        if state["documents"]:
            return "generate"
        if state.get("rewrites", 0) < config.MAX_REWRITES:
            return "rewrite_query"
        return "not_found"

    g = StateGraph(GraphState)
    g.add_node("route", route_node)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("generate", generate)
    g.add_node("off_topic", off_topic)
    g.add_node("not_found", not_found)

    g.add_edge(START, "route")
    g.add_conditional_edges("route", decide_route)
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges("grade_documents", decide_after_grading)
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("generate", END)
    g.add_edge("off_topic", END)
    g.add_edge("not_found", END)
    return g.compile()