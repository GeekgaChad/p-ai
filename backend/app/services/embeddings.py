#bedrock adapter (titan v2) for embeddings

import boto3, json
from ..config import settings

bedrock = boto3.client("bedrock-runtime", region_name=settings.AWS_REGION)

def embed_texts(texts: list[str]) -> list[list[float]]:
    resp = bedrock.invoke_model(
        modelId=settings.BEDROCK_EMBED_MODEL,
        body=json.dumps({"inputText": texts}) if isinstance(texts, str) else json.dumps({"inputText": texts}),
        accept="application/json",
        contentType="application/json",
    )
    payload = json.loads(resp["body"].read())
    # Titan v2 returns {"embedding":[...]} for single; some SDKs return list—normalize:
    if "embedding" in payload:
        return [payload["embedding"]]
    if "embeddings" in payload:
        return payload["embeddings"]
    raise RuntimeError("Unknown embed response")
