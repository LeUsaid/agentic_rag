"""LangGraph agentic RAG workflow (document-only).

START -> route -+-> greeting -> END
                +-> off_topic -> END
                +-> contextualize -> retrieve -> grade_documents -+-> generate -> END
                       ^                                          |
                       |                                          +-> rewrite_query (retries left)
                       +------------------------------------------+
                                                                  +-> not_found -> END
"""
from typing import List, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

import config
import prompts

OFF_TOPIC_MESSAGE = (
    "This question isn't related to the uploaded document, so I can't answer it. "
    "Please ask something about the document."
)
NOT_FOUND_MESSAGE = "I couldn't find relevant information about that in the document."


class RouteDecision(BaseModel):
    datasource: Literal["document", "greeting", "off_topic"] = Field(
        description="Where to send the question"
    )


class Grade(BaseModel):
    relevant: Literal["yes", "no"] = Field(description="Is the passage relevant?")


class GraphState(TypedDict, total=False):
    question: str
    query: str
    documents: List[Document]
    generation: str
    rewrites: int
    history: List[BaseMessage]


def _chat(system: str, human: str, with_history: bool = False) -> ChatPromptTemplate:
    messages: list = [("system", system)]
    if with_history:
        messages.append(MessagesPlaceholder("history", optional=True))
    messages.append(("human", human))
    return ChatPromptTemplate.from_messages(messages)


def build_graph(vectorstore, api_key: str, summary: str):
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL, google_api_key=api_key, temperature=0
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})

    router = _chat(prompts.ROUTER_SYSTEM, prompts.ROUTER_HUMAN, with_history=True) | llm.with_structured_output(
        RouteDecision
    )
    grader = _chat(prompts.GRADER_SYSTEM, prompts.GRADER_HUMAN, with_history=True) | llm.with_structured_output(
        Grade
    )
    rewriter = _chat(prompts.REWRITE_SYSTEM, prompts.REWRITE_HUMAN, with_history=True) | llm | StrOutputParser()
    contextualizer = (
        _chat(prompts.CONTEXTUALIZE_SYSTEM, prompts.CONTEXTUALIZE_HUMAN, with_history=True)
        | llm
        | StrOutputParser()
    )
    generator = _chat(prompts.GENERATE_SYSTEM, prompts.GENERATE_HUMAN, with_history=True) | llm | StrOutputParser()
    greeter = _chat(prompts.GREETING_SYSTEM, prompts.GREETING_HUMAN, with_history=True) | llm | StrOutputParser()

    def _history(state: GraphState) -> list:
        return state.get("history") or []

    def route_node(state: GraphState):
        return {"query": state["question"], "rewrites": 0, "documents": []}

    def contextualize(state: GraphState):
        history = _history(state)
        if not history:
            return {"query": state["question"]}
        query = contextualizer.invoke({"question": state["question"], "history": history})
        return {"query": query.strip()}

    def retrieve(state: GraphState):
        return {"documents": retriever.invoke(state["query"])}

    def grade_documents(state: GraphState):
        history = _history(state)
        inputs = [
            {"question": state["question"], "document": d.page_content, "history": history}
            for d in state["documents"]
        ]
        grades = grader.batch(inputs)
        kept = [d for d, g in zip(state["documents"], grades) if g.relevant == "yes"]
        return {"documents": kept}

    def rewrite_query(state: GraphState):
        new_query = rewriter.invoke(
            {"question": state["question"], "history": _history(state)}
        )
        return {"query": new_query.strip(), "rewrites": state.get("rewrites", 0) + 1}

    def generate(state: GraphState):
        context = "\n\n".join(d.page_content for d in state["documents"])
        answer = generator.invoke(
            {
                "context": context,
                "question": state["question"],
                "history": _history(state),
            }
        )
        return {"generation": answer}

    def greeting(state: GraphState):
        text = greeter.invoke({"question": state["question"], "history": _history(state)})
        return {"generation": text, "documents": []}

    def off_topic(state: GraphState):
        return {"generation": OFF_TOPIC_MESSAGE, "documents": []}

    def not_found(state: GraphState):
        return {"generation": NOT_FOUND_MESSAGE, "documents": []}

    def decide_route(state: GraphState) -> Literal["contextualize", "greeting", "off_topic"]:
        decision = router.invoke(
            {
                "question": state["question"],
                "summary": summary,
                "history": _history(state),
            }
        )
        if decision.datasource == "document":
            return "contextualize"
        if decision.datasource == "greeting":
            return "greeting"
        return "off_topic"

    def decide_after_grading(state: GraphState) -> Literal["generate", "rewrite_query", "not_found"]:
        if state["documents"]:
            return "generate"
        if state.get("rewrites", 0) < config.MAX_REWRITES:
            return "rewrite_query"
        return "not_found"

    g = StateGraph(GraphState)
    g.add_node("route", route_node)
    g.add_node("contextualize", contextualize)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("generate", generate)
    g.add_node("greeting", greeting)
    g.add_node("off_topic", off_topic)
    g.add_node("not_found", not_found)

    g.add_edge(START, "route")
    g.add_conditional_edges("route", decide_route)
    g.add_edge("contextualize", "retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges("grade_documents", decide_after_grading)
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("generate", END)
    g.add_edge("greeting", END)
    g.add_edge("off_topic", END)
    g.add_edge("not_found", END)
    return g.compile()
