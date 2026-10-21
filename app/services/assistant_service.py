"""
Assemble le contexte RAG (rag_store), les données métier temps réel
(django_client) et l'historique de la conversation, puis passe par llm_router
(chaîne de modèles configurée avec secours inter-fournisseurs).
"""

import json
import logging
from datetime import date
from typing import Dict, List, Optional

from app.integrations.django_client import get_live_context
from app.services.llm_router import llm_router
from app.services.rag_store import rag_store

logger = logging.getLogger(__name__)

MAX_CHARS_DONNEES_LIVE = 8000
MAX_CHARS_DOCUMENTS = 6000
MAX_CHARS_QUESTION = 4000
MAX_CHARS_HISTORIQUE = 2400
MARQUEUR_TRONCATURE = "[Contexte partiel : le contenu a été tronqué pour respecter la taille maximale.]"

_SYSTEM_PROMPT = (
    "Tu es l'assistant EcoScan, conseiller en efficacité énergétique pour des organisations "
    "au Sénégal (Senelec, compteurs Woyofal, montants en FCFA ; ne suppose ni chauffage ni climatisation sans donnée "
    "disponible). Date du jour : {jour}. Réponds en français, de façon claire et concise "
    "(moins de 150 mots, sauf demande contraire).\n"
    "Règles : appuie-toi UNIQUEMENT sur le contexte documentaire et les données métier fournis "
    "ci-dessous ; n'invente ni équipements, ni personnes, ni rôles, ni délais. Si une donnée "
    "manque, indique où la saisir uniquement si ce parcours est confirmé ci-dessous. Capacités "
    "mobiles confirmées : « Plus » > « Relevés et factures » dépose des documents au niveau de "
    "l'organisation, sans OCR, analyse ni saisie manuelle des valeurs. L'application mobile ne "
    "propose pas de saisie ou d'import des consommations par équipement. N'invente aucun autre "
    "formulaire ou parcours. Si la question nécessite ces valeurs, dis explicitement que l'app "
    "ne permet pas de les saisir actuellement et qu'une intégration de sous-compteurs ou un "
    "nouveau parcours de saisie est nécessaire. Ne recommande pas de déposer un document comme "
    "moyen d'obtenir automatiquement ces mesures, et ne dis jamais qu'elles pourront être saisies "
    "manuellement après le dépôt. Une hypothèse est "
    "toujours PROBABLE, jamais certaine. Pour un objectif, "
    "« declaree » = économies estimées des recommandations décidées, « mesuree » = calculée sur "
    "les données réelles : ne les confonds jamais, cite la source et la fiabilité de la mesure, et "
    "si « mesuree » est vide, donne la « raison ». Pour une anomalie, propose de confirmer "
    "l'hypothèse puis de créer une recommandation uniquement si le parcours est confirmé dans "
    "les données fournies.\n"
    "Pour une comparaison, ne compare que des mesures de même unité et de périodes comparables ; "
    "donne les valeurs, les dates, l'écart absolu et le pourcentage uniquement si les données le "
    "permettent. Pour une prévision de consommation, utilise d'abord une prévision explicite du "
    "contexte ; sinon calcule une moyenne des variations d'index journalières VALIDEE du même "
    "compteur, seulement avec au moins trois journées consécutives. Indique l'horizon, la méthode "
    "et le nombre de jours observés ; présente le résultat comme une estimation, jamais comme une "
    "certitude. Si les relevés ne suffisent pas, dis-le et ne fabrique aucun chiffre. Les ordres "
    "matériels sont envoyés séparément par l'application : ne prétends jamais qu'un équipement a "
    "été allumé ou arrêté sans statut de commande CONFIRMED dans les données fournies. Termine "
    "par UNE prochaine étape concrète. Si le contexte JSON contient `_tronque`, "
    "il est incomplet : ne traite jamais une information absente de l'extrait comme une preuve."
)


def _system_prompt() -> str:
    # Calculé à chaque requête : une constante figerait la date au démarrage du service.
    return _SYSTEM_PROMPT.format(jour=date.today().strftime("%d/%m/%Y"))


def _formater_historique(history: Optional[List[Dict[str, str]]]) -> str:
    lignes = []
    caracteres_restants = MAX_CHARS_HISTORIQUE
    for h in (history or [])[-4:]:
        role = "Assistant" if h.get("role") == "assistant" else "Utilisateur"
        prefixe = f"{role} : "
        separateur = 1 if lignes else 0
        contenu_max = min(600, caracteres_restants - len(prefixe) - separateur)
        if contenu_max <= 0:
            break
        contenu = str(h.get("content", "")).strip()[:contenu_max]
        if contenu:
            ligne = f"{prefixe}{contenu}"
            lignes.append(ligne)
            caracteres_restants -= len(ligne) + separateur
        if caracteres_restants <= 0:
            break
    return "\n".join(lignes) or "Aucun échange précédent."


def _serialiser_contexte_live(contexte: dict) -> str:
    texte = json.dumps(contexte, ensure_ascii=False, default=str, separators=(",", ":"))
    if len(texte) <= MAX_CHARS_DONNEES_LIVE:
        return texte

    extrait = texte[:MAX_CHARS_DONNEES_LIVE]
    encadre = json.dumps(
        {"_tronque": True, "extrait_json_partiel": extrait},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    while len(encadre) > MAX_CHARS_DONNEES_LIVE:
        extrait = extrait[: max(0, len(extrait) - (len(encadre) - MAX_CHARS_DONNEES_LIVE))]
        encadre = json.dumps(
            {"_tronque": True, "extrait_json_partiel": extrait},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return encadre


async def repondre(
    question: str,
    organisation_id: str,
    limit: int = 5,
    include_live_data: bool = True,
    history: Optional[List[Dict[str, str]]] = None,
) -> dict:
    passages = rag_store.search(question, organisation_id, limit=limit)
    contexte_documentaire = "\n\n".join(
        f"Source: {p['source']}\n{p['text']}" for p in passages
    )
    if len(contexte_documentaire) > MAX_CHARS_DOCUMENTS:
        contexte_documentaire = (
            contexte_documentaire[: MAX_CHARS_DOCUMENTS - len(MARQUEUR_TRONCATURE) - 1]
            + "\n"
            + MARQUEUR_TRONCATURE
        )
    if not contexte_documentaire:
        contexte_documentaire = "Aucun document pertinent trouvé pour cette organisation."

    donnees_live = "Aucune donnée métier en temps réel demandée."
    if include_live_data:
        contexte_live = await get_live_context(organisation_id)
        if "_error" in contexte_live:
            logger.warning("Contexte Django indisponible : %s", contexte_live["_error"])
            donnees_live = f"Données temps réel indisponibles ({contexte_live['_error']})."
        else:
            donnees_live = _serialiser_contexte_live(contexte_live)

    question_tronquee = len(question) > MAX_CHARS_QUESTION
    question_contexte = question[:MAX_CHARS_QUESTION]
    if question_tronquee:
        marqueur = " [question tronquée]"
        question_contexte = question[: MAX_CHARS_QUESTION - len(marqueur)] + marqueur
    prompt = (
        f"Historique de la conversation :\n{_formater_historique(history)}\n\n"
        f"Question : {question_contexte}\n\n"
        f"Contexte documentaire :\n{contexte_documentaire}\n\n"
        f"Données actuelles de l'organisation (JSON) :\n{donnees_live}"
    )

    reponse_texte, modele_utilise = await llm_router.generate_text(
        prompt=prompt,
        system_prompt=_system_prompt(),
    )

    return {"answer": reponse_texte, "sources": passages, "model_used": modele_utilise}
