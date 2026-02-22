def build_rag_prompt(context_chunks, user_query):
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a highly knowledgeable medical AI assistant.

Use ONLY the provided medical document context to answer.

Medical Document Context:
-------------------------
{context}
-------------------------

User Question:
{user_query}

Provide:
- Clear medical reasoning
- Risk level if applicable (LOW / MODERATE / HIGH)
- Avoid making final diagnosis
- Add safety disclaimer
"""

    return prompt