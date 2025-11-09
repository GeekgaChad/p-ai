# Bedrock LLM call + prompt

import boto3, json
from ..config import settings

session = boto3.Session(
    region_name=settings.BEDROCK_REGION or settings.AWS_REGION,
    profile_name=settings.AWS_PROFILE
)
chat = session.client("bedrock-runtime")

SYSTEM_PROMPT = (
    "You are a precise study assistant. Answer ONLY using the provided excerpts. "
    "Cite them as [Title #chunk]. If unsure, say you don't know."
)

def format_prompt(query: str, passages: list[tuple[str, str, int]]):
    ctx = "\n\n".join([f"[{t} #{i}] {x}" for (t, x, i) in passages])
    return f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{ctx}\n\nQUESTION: {query}"

def generate_answer(query: str, passages: list[tuple[str, str, int]]) -> str:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 600,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": [{"type": "text", "text": format_prompt(query, passages)}]}],
    }
    resp = chat.invoke_model(
        modelId=settings.BEDROCK_CHAT_MODEL,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )
    data = json.loads(resp["body"].read())
    return data["content"][0]["text"]

