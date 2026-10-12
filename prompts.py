ROUTER_PROMPT = """You are a strict router for a Q&A assistant that ONLY answers questions about one uploaded document.

Here is a preview of the document:
---
{preview}
---

Decide where to send the user's question.
- "document": the question could plausibly be answered by this document, or is about its topic.
- "off_topic": anything else (general knowledge, how-to questions, news, coding help, small talk, greetings, or any subject not covered by the document).

Question: {question}"""

GRADER_PROMPT = """You are grading whether a retrieved passage is relevant to a question.
Answer "yes" if the passage contains information useful for answering the question, otherwise "no".

Question: {question}

Passage:
{document}"""

REWRITE_PROMPT = """The question below did not retrieve relevant passages from a document.
Rewrite it as a clearer, more specific search query that is likely to match the document's wording.
Return only the rewritten query.

Original question: {question}"""

GENERATE_PROMPT = """You are a helpful assistant answering questions using ONLY the context below.
If the context does not contain the answer, say you could not find it in the document.
Be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""

DIRECT_PROMPT = """You are a friendly assistant for a document Q&A app. Reply briefly to:
{question}"""
