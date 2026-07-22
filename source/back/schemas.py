from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str


class ClassifyResponse(BaseModel):
    animal: str
    conf: float


class FindNearestResponse(BaseModel):
    image: str  # base64-encoded data 
    similarity: float