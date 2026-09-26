import httpx

from app.core.config import settings


async def get_live_context(organisation_id: str) -> dict:
    """Récupère le contexte métier temps réel côté Django pour une organisation.

    Endpoint : analysis/internal_views.py::AIContextView, monté sous
    /api/internal/ (voir analysis/internal_urls.py). Le jeton envoyé ici
    (DJANGO_INTERNAL_TOKEN) doit être identique à settings.DJANGO_INTERNAL_TOKEN
    côté Django.
    """
    url = f"{settings.DJANGO_API_URL.rstrip('/')}/api/internal/organisations/{organisation_id}/ai-context/"
    headers = {"X-Internal-Service-Token": settings.DJANGO_INTERNAL_TOKEN}
    try:
        async with httpx.AsyncClient(timeout=settings.DJANGO_TIMEOUT) as client:
            r = await client.get(url, headers=headers)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPStatusError as exc:
        return {"_error": f"Django a répondu HTTP {exc.response.status_code} sur ai-context"}
    except (httpx.HTTPError, ValueError) as exc:
        return {"_error": f"Django indisponible : {exc}"}
    return data if isinstance(data, dict) else {"_error": "Réponse Django inattendue (objet JSON attendu)."}
