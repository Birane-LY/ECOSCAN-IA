import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from app.core.config import settings

api_key_header = APIKeyHeader(name="X-Internal-Service-Token", auto_error=False)

async def verify_internal_token(token: str = Security(api_key_header)) -> str:
    """
    Vérifie que la requête HTTP contient un jeton de sécurité interne valide.
    
    Ce mécanisme protège les endpoints internes de FastAPI pour s'assurer
    que seul le backend Django (ou un service autorisé) peut y accéder.
    
    """
    if not settings.INTERNAL_TOKEN or settings.INTERNAL_TOKEN == "change-me-in-production":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="INTERNAL_TOKEN non configuré en production.",
        )

    if not token or not secrets.compare_digest(token, settings.INTERNAL_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Jeton de sécurité interne absent ou invalide.",
            headers={"WWW-Authenticate": "X-Internal-Service-Token"},
        )
    return token