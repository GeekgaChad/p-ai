# Upload bytes to S3 and return the S3 URI

import boto3, uuid
from ..config import settings
s3 = boto3.client("s3", region_name=settings.AWS_REGION)

def put_bytes(filename: str, data: bytes, mime: str) -> str:
    key = f"uploads/{uuid.uuid4()}-{filename}"
    s3.put_object(Bucket=settings.S3_BUCKET, Key=key, Body=data, ContentType=mime)
    return f"s3://{settings.S3_BUCKET}/{key}"
