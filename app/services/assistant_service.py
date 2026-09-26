"""
Assemble le contexte RAG (rag_store), les données métier temps réel
(django_client) et l'historique de la conversation, puis passe par llm_router
(chaîne de secours Groq/HuggingFace/OpenAI).
"""

import json
import logging
from datetime import date
from typing import Dict, List, Optional

from app.integrations.django_client import get_live_context
from app.services.llm_router import llm_router
from app.services.rag_store import rag_store

logger = logging.getLogger(__name__)

MAX_CHARS_DONNEES_LIVE = 14000

_SYSTEM_PROMPT = (
    "Tu es l'assistant EcoScan, conseiller en efficacité énergétique pour des organisations "
    "au Sénégal (Senelec, compteurs Woyofal, montants en FCFA ; ne suppose ni chauffage ni climatisation sans donnée "
    "disponible). Date du jour : {jour}. Réponds en français, de façon claire et concise "
    "(moins de 150 mots, sauf demande contraire).\n"
    "Règles : appuie-toi UNIQUEMENT sur le contexte documentaire et les données métier fournis "
    "ci-dessous ; n'invente ni équipements, ni personnes, ni rôles, ni délais. Si une donnée "
    "manque, dis laquelle et où la saisir dans EcoScan (import de facture, relevé de solde, "
    "objectif). Une hypothèse est toujours PROBABLE, jamais certaine. Pour un objectif, "
    "« declaree » = économies estimées des recommandations décidées, « mesuree » = calculée sur "
    "les données réelles : ne les confonds jamais, cite la source et la fiabilité de la mesure, et "
    "si « mesuree » est vide, donne la « raison ». Pour une anomalie, propose de confirmer "
    "l'hypothèse puis de créer une recommandation depuis la page Analyses. Termine par UNE "
    "prochaine étape concrète."
)


def _system_prompt() -> str:
    # Calculé à chaque requête : une constante figerait la date au démarrage du service.
    return _SYSTEM_PROMPT.format(jour=date.today().strftime("%d/%m/%Y"))


def _formater_historique(history: Optional[List[Dict[str, str]]]) -> str:
    lignes = []
    for h in (history or [])[-6:]:
        role = "Assistant" if h.get("role") == "assistant" else "Utilisateur"
        contenu = str(h.get("content", "")).strip()[:1500]
        if contenu:
            lignes.append(f"{role} : {contenu}")
    return "\n".join(lignes) or "Aucun échange précédent."


async def repondre(
    question: str,
    organisation_id: str,
    limit: int = 5,
    include_live_data: bool = True,
    history: Optional[List[Dict[str, str]]] = None,
) -> dict:
    passages = rag_store.search(question, organisation_id, limit=limit)
    contexte_documentaire = (
        "\n\n".join(f"Source: {p['source']}\n{p['text']}" for p in passages)
        or "Aucun document pertinent trouvé pour cette organisation."
    )

    donnees_live = "Aucune donnée métier en temps réel demandée."
    if include_live_data:
        contexte_live = await get_live_context(organisation_id)
        if "_error" in contexte_live:
            logger.warning("Contexte Django indisponible : %s", contexte_live["_error"])
            donnees_live = f"Données temps réel indisponibles ({contexte_live['_error']})."
        else:
            donnees_live = json.dumps(contexte_live, ensure_ascii=False, default=str, separators=(",", ":"))[:MAX_CHARS_DONNEES_LIVE]

    prompt = (
        f"Historique de la conversation :\n{_formater_historique(history)}\n\n"
        f"Question : {question}\n\n"
        f"Contexte documentaire :\n{contexte_documentaire}\n\n"
        f"Données actuelles de l'organisation (JSON) :\n{donnees_live}"
    )

    try:
        reponse_texte, modele_utilise = await llm_router.generate_text(prompt=prompt, system_prompt=_system_prompt())
    except RuntimeError as exc:
        logger.error("Chaîne de secours LLM entièrement épuisée : %s", exc)
        reponse_texte = "Le service IA est temporairement indisponible — merci de réessayer dans un instant."
        modele_utilise = "none"

    return {"answer": reponse_texte, "sources": passages, "model_used": modele_utilise}
