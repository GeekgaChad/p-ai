#bedrock adapter (titan v2) for embeddings

import boto3, json
from ..config import settings

_region = settings.BEDROCK_REGION or settings.AWS_REGION
bedrock = boto3.client("bedrock-runtime", region_name=_region)

def embed_texts(texts: list[str]) -> list[list[float]]:
    payload = {"inputText": texts} if isinstance(texts, list) else {"inputText": [texts]}
    resp = bedrock.invoke_model(
        modelId=settings.BEDROCK_EMBED_MODEL,
        body=json.dumps(payload),
        accept="application/json",
        contentType="application/json",
    )
    data = json.loads(resp["body"].read())
    # Titan returns either "embedding" (single) or "embeddings" (batch), handle both:
    if "embedding" in data:
        return [data["embedding"]]
    if "embeddings" in data:
        return data["embeddings"]
    raise RuntimeError(f"Unexpected embed response: keys={list(data.keys())}")



