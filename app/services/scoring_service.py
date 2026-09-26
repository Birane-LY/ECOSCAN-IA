import json
import logging
import time
from typing import Dict, Tuple

from groq import AsyncGroq
from app.core.config import settings
from app.core.contracts import ClaimCategory, PriorityLevel

logger = logging.getLogger(__name__)

DEFAULT_SCORING_MODEL = "openai/gpt-oss-120b"


class ScoringServiceError(Exception):
    """Raised when scoring output cannot be parsed."""


class ScoringService:
    def __init__(self):
        self._client = None

    @property
    def client(self) -> AsyncGroq:
        if self._client is None:
            self._client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        return self._client

    async def analyze_claim(self, text: str) -> Tuple[Dict, float]:
        """
        Analyzes claim text for sentiment, priority scoring, and classification.
        """
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not configured.")

        start_time = time.time()

        categories = [c.value for c in ClaimCategory]
        priorities = [p.value for p in PriorityLevel]

        prompt = (
            "Tu es un système expert d'analyse des réclamations pour la Senelec (électricité).\n"
            "Analyse le texte de réclamation fourni et extrais la catégorie, la priorité, "
            "le sentiment, le score d'urgence et une recommandation.\n\n"
            f"Catégories autorisées : {categories}\n"
            f"Niveaux de priorité autorisés : {priorities}\n\n"
            "Règles d'évaluation :\n"
            "- CRITICAL / Urgency > 0.8 : Câbles dénudés/tombés, danger d'électrocution, coupure générale prolongée, code Woyofal payé mais non reçu en situation d'urgence.\n"
            "- HIGH / Urgency 0.6-0.8 : Compteur en panne, surfacturation manifeste, coupure ciblée non planifiée.\n"
            "- MEDIUM / Urgency 0.3-0.6 : Erreur d'index, demande de clarification sur facture.\n"
            "- LOW / Urgency < 0.3 : Demande d'information générale, démarche administrative standard.\n\n"
            "Format de réponse exigé (STRICT JSON) :\n"
            "{\n"
            '  "category": "<UNE_DES_CATEGORIES_AUTORISEES>",\n'
            '  "priority": "<UN_DES_NIVEAUX_DE_PRIORITE>",\n'
            '  "sentiment": "<sentiment_décelé>",\n'
            '  "urgency_score": 0.85,\n'
            '  "summary": "<résumé synthétique en une phrase>",\n'
            '  "action_recommended": "<action conseillée au support>"\n'
            "}\n\n"
            f"Texte de la réclamation : \"{text}\""
        )

        try:
            response = await self.client.chat.completions.create(
                model=DEFAULT_SCORING_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            content_str = response.choices[0].message.content
        except Exception as exc:
            raise ScoringServiceError(f"Error calling Groq API for scoring: {exc}") from exc

        execution_time = round(time.time() - start_time, 3)

        try:
            parsed = json.loads(content_str)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ScoringServiceError(f"Invalid JSON from scoring model: {exc}") from exc

        return parsed, execution_time


scoring_service = ScoringService()