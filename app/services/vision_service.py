import base64
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from groq import AsyncGroq

from app.core.config import settings

logger = logging.getLogger(__name__)

TAILLE_MAX_IMAGE_OCTETS = 20 * 1024 * 1024
MODELE_VISION_PAR_DEFAUT = "qwen/qwen3.6-27b"
# La limite OTPM du tier actuel est de 1 000 tokens.
MAX_COMPLETION_TOKENS = 700

CHAMPS_FACTURE = (
    "numero_facture",
    "numero_partenaire",
    "police",
    "numero_compteur",
    "numero_compte",
    "date_facture",
    "ancien_index",
    "nouveau_index",
    "consommation_kwh",
    "montant_net_paye",
    "unite",
    "devise",
)


class VisionExtractionError(Exception):
    """Erreur levée lorsque l'extraction de l'image échoue."""


class VisionService:
    def __init__(self) -> None:
        api_key = settings.GROQ_API_KEY
        if not api_key:
            raise ValueError(
                "La variable GROQ_API_KEY n'est pas configurée."
            )

        # Ne pas définir base_url : le SDK ajoute automatiquement
        # /openai/v1/chat/completions à l'URL officielle Groq.
        self.client = AsyncGroq(api_key=api_key)

    async def analyser_facture_ou_compteur(
        self,
        image_bytes: bytes,
        mime_type: str,
        modele: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], List[str], float]:
        """Analyse une facture ou un compteur et extrait les champs attendus."""

        if not settings.GROQ_API_KEY:
            raise VisionExtractionError(
                "La variable GROQ_API_KEY n'est pas configurée."
            )

        if not image_bytes:
            raise VisionExtractionError("L'image reçue est vide.")

        if len(image_bytes) > TAILLE_MAX_IMAGE_OCTETS:
            raise VisionExtractionError(
                f"Image trop volumineuse ({len(image_bytes)} octets). "
                f"La limite Groq est de {TAILLE_MAX_IMAGE_OCTETS} octets."
            )

        if not mime_type or not mime_type.startswith("image/"):
            raise VisionExtractionError(
                f"Type MIME d'image invalide : {mime_type!r}."
            )

        start_time = time.time()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{base64_image}"

        champs_liste = ", ".join(CHAMPS_FACTURE)
        prompt = (
            "Lis cette facture énergétique Senelec ou ce compteur électrique. "
            f"Retourne uniquement un objet JSON avec les clés champs et "
            f"champs_non_lisibles. Dans champs, utilise uniquement ces clés : "
            f"{champs_liste}.\n"
            "Pour toute valeur illisible, utilise null et ajoute sa clé à "
            "champs_non_lisibles. Ne devine rien. Les index et la consommation "
            "sont des entiers. La date reste au format JJ/MM/AAAA. "
            "Réponse JSON uniquement, sans Markdown ni explication."
        )

        try:
            response = await self.client.chat.completions.create(
                model=modele or MODELE_VISION_PAR_DEFAUT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                        ],
                    }
                ],
                temperature=0.0,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                reasoning_effort="none",
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            logger.exception("Erreur détaillée lors de l'appel à l'API Groq")
            raise VisionExtractionError(
                "Erreur lors de l'appel à l'API Groq "
                f"({type(exc).__name__}) : {exc}"
            ) from exc

        execution_time = round(time.time() - start_time, 3)

        try:
            content_str = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise VisionExtractionError(
                "La réponse Groq ne contient aucun choix exploitable."
            ) from exc

        if not content_str:
            raise VisionExtractionError(
                "Le modèle Groq a renvoyé une réponse vide."
            )

        try:
            parsed = json.loads(content_str)
        except json.JSONDecodeError as exc:
            raise VisionExtractionError(
                "Réponse du modèle non exploitable en JSON : "
                f"{exc}. Réponse reçue : {content_str[:500]!r}"
            ) from exc

        if not isinstance(parsed, dict):
            raise VisionExtractionError(
                "La réponse JSON du modèle doit être un objet."
            )

        champs_bruts = parsed.get("champs", {})
        if not isinstance(champs_bruts, dict):
            champs_bruts = {}

        champs_extraits = {
            cle: champs_bruts.get(cle) for cle in CHAMPS_FACTURE
        }

        champs_non_lisibles = parsed.get("champs_non_lisibles", [])
        if not isinstance(champs_non_lisibles, list):
            champs_non_lisibles = []
        else:
            champs_non_lisibles = [
                cle for cle in champs_non_lisibles
                if isinstance(cle, str) and cle in CHAMPS_FACTURE
            ]

        return champs_extraits, champs_non_lisibles, execution_time

    async def fermer(self) -> None:
        """Ferme proprement le client HTTP Groq."""
        close_method = getattr(self.client, "close", None)
        if close_method is not None:
            result = close_method()
            if result is not None:
                await result


vision_service = VisionService()
