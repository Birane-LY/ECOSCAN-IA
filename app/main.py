from fastapi import FastAPI
from app.api.v1.endpoints import router as api_v1_router
from app.core.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="Microservice IA/RAG pour EcoScan"
)

# Inclusion des routes v1
app.include_router(api_v1_router, prefix="/api/v1")


@app.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "ok", "app": settings.APP_NAME, "environment": settings.ENVIRONMENT}