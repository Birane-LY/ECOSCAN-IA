import secrets

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

    Deux durcissements par rapport à la version précédente :
    - secrets.compare_digest au lieu de '!=' : une comparaison de chaînes
      standard s'arrête au premier caractère différent, ce qui fuit
      (très légèrement) la longueur du préfixe correct via le temps de
      réponse. Négligeable en pratique ici, mais c'est le même réflexe que
      celui déjà appliqué à ce jeton côté Django (security.py de ce projet).
    - Un jeton laissé à sa valeur par défaut ("change-me-in-production") est
      désormais explicitement rejeté avec un 500, au lieu de fonctionner
      silencieusement en acceptant n'importe quelle requête qui présente
      elle-même cette même valeur par défaut.

    Args:
        token (str): Le jeton d'API extrait de l'en-tête 'X-Internal-Token'.

    Returns:
        str: Le jeton validé en cas de succès.

    Raises:
        HTTPException: 500 si INTERNAL_TOKEN n'a jamais été configuré, 401 si
            le jeton présenté est absent ou invalide.
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
            headers={"WWW-Authenticate": "X-Internal-Token"},
        )
    return token