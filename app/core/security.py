from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from app.core.config import settings

# En-tête HTTP personnalisé utilisé pour transporter le jeton interne
api_key_header = APIKeyHeader(name="X-Internal-Token", auto_error=False)


async def verify_internal_token(token: str = Security(api_key_header)) -> str:
    """
    Vérifie que la requête HTTP contient un jeton de sécurité interne valide.

    Ce mécanisme protège les endpoints internes de FastAPI pour s'assurer
    que seul le backend Django (ou un service autorisé) peut y accéder.

    Args:
        token (str): Le jeton d'API extrait de l'en-tête 'X-Internal-Token'.

    Returns:
        str: Le jeton validé en cas de succès.

    Raises:
        HTTPException: Erreur 401 (UNAUTHORIZED) si le jeton est manquant ou invalide.
    """
    if not token or token != settings.INTERNAL_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Jeton de sécurité interne absent ou invalide.",
            headers={"WWW-Authenticate": "X-Internal-Token"},
        )
    return token