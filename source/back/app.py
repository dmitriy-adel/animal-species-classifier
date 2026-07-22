"""
Ручки:
- GET  /health        -> {"status": "ok"}
- POST /classify       -> {"animal": str, "conf": float}
- POST /find_nearest    -> {"image": str, "similarity": float}
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from model_manager import ModelManager
from records_base import RecordsBase
from tools import Tools
from schemas import HealthResponse, ClassifyResponse, FindNearestResponse

MODEL_PATH = os.getenv("MODEL_PATH", "../../models/")
RECORDS_BASE_PATH = os.getenv("RECORDS_BASE_PATH", "data/")

mm: ModelManager = ModelManager()
rb: RecordsBase = RecordsBase()
tls: Tools = Tools()


@asynccontextmanager
async def lifespan(app: FastAPI):
    mm.initialize(MODEL_PATH, model_filename='best_model-for_only_inaturalist.pt')
    rb.initialize(RECORDS_BASE_PATH, model_manager=mm)
    yield


app = FastAPI(title="Animal Classifier API", lifespan=lifespan)


async def _read_uploaded_image(file: UploadFile):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")
        
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="an empty file")
        
    try:
        return tls.bytes_to_image(raw)
        
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/classify", response_model=ClassifyResponse)
async def classify(file: UploadFile = File(...)) -> ClassifyResponse:
    image = await _read_uploaded_image(file)
    try:
        label, confidence = mm.classify(image)
        
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
        
    return ClassifyResponse(animal=label, conf=confidence)


@app.post("/find_nearest", response_model=FindNearestResponse)
async def find_nearest(file: UploadFile = File(...)) -> FindNearestResponse:
    image = await _read_uploaded_image(file)

    try:
        query_vector = mm.vectorize(image)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        nearest_image, similarity = rb.find_nearest(query_vector)
        
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
        
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    encoded_image = tls.image_to_base64(nearest_image)
    return FindNearestResponse(image=encoded_image, similarity=similarity)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})
