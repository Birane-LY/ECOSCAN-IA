from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


# --- Schémas pour l'Inférence Texte / RAG ---

class RAGQueryRequest(BaseModel):
    """
    Modèle de requête pour les interrogations texte et RAG.
    """
    query: str = Field(
        ...,
        min_length=1,
        description="Question ou instruction soumise au modèle."
    )
    use_rag: bool = Field(
        default=True,
        description="Active ou désactive la recherche contextuelle RAG."
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Nombre maximal de documents pertinents à extraire pour le RAG."
    )
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Historique optionnel de la conversation (ex: [{'role': 'user', 'content': '...'}])."
    )


class RAGQueryResponse(BaseModel):
    """
    Modèle de réponse pour les traitements texte et RAG.
    """
    answer: str = Field(
        ...,
        description="Réponse générée par le modèle d'IA."
    )
    sources: List[Dict[str, Any]] = Field(
        default=[],
        description="Liste des extraits de documents utilisés pour élaborer la réponse."
    )
    model_used: str = Field(
        ...,
        description="Identifiant du modèle ayant traité la requête après résolution de la chaîne de secours."
    )
    execution_time_seconds: float = Field(
        ...,
        description="Temps d'exécution total du traitement en secondes."
    )


# --- Schémas pour la Vision / Analyse d'Image ---

class VisionAnalysisRequest(BaseModel):
    """
    Modèle de requête pour l'analyse d'image.
    """
    image_url: str = Field(
        ...,
        description="URL de l'image à analyser."
    )
    prompt: Optional[str] = Field(
        default="Décris et analyse cette image en détail.",
        description="Question ou consigne spécifique concernant l'image."
    )


class VisionAnalysisResponse(BaseModel):
    """
    Modèle de réponse pour l'analyse d'image.
    """
    description: str = Field(..., description="Résultat textuel de l'analyse multimodale.")
    champs: Dict[str, Any] = Field(default_factory=dict, description="Champs structurés extraits de l'image.")
    champs_non_lisibles: List[str] = Field(default_factory=list, description="Clés dont la valeur n'a pas pu être lue.")
    model_used: str = Field(..., description="Identifiant du modèle de vision utilisé.")
    execution_time_seconds: float = Field(..., description="Temps d'exécution total en secondes.")


# --- Schémas pour la Transcription Audio ---

class AudioTranscriptionRequest(BaseModel):
    """
    Modèle de requête pour la transcription audio.
    """
    audio_url: str = Field(
        ...,
        description="URL du fichier audio à transcrire."
    )
    language: Optional[str] = Field(
        default=None,
        description="Code langue optionnel (ex: 'fr', 'wo' pour Wolof)."
    )


class AudioTranscriptionResponse(BaseModel):
    """
    Modèle de réponse pour la transcription audio.
    """
    transcription: str = Field(
        ...,
        description="Texte intégral transcrit à partir de l'audio."
    )
    language_detected: Optional[str] = Field(
        default=None,
        description="Langue détectée par le moteur de transcription."
    )
    engine_used: str = Field(
        ...,
        description="Moteur de transcription utilisé (ex: 'faster-whisper')."
    )
    execution_time_seconds: float = Field(
        ...,
        description="Temps d'exécution total en secondes."
    )



class ImageAnalysisRequest(BaseModel):
    candidate_labels: List[str] = Field(
        default=["colis endommagé", "reçu / facture", "produit défectueux", "erreur de livraison", "autre"],
        description="Liste de catégories pour la classification Zero-Shot"
    )

class ImageAnalysisResponse(BaseModel):
    summary: str = Field(description="Description visuelle détaillée ou résumé de l'image")
    primary_label: str = Field(description="Catégorie principale détectée")
    confidence: float = Field(description="Score de confiance pour la catégorie principale (0 à 1)")
    all_scores: dict = Field(default_factory=dict, description="Distribution des scores par catégorie")
    execution_time_seconds: float = Field(description="Temps d'exécution en secondes")


class PriorityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ClaimCategory(str, Enum):
    BILLING = "BILLING"                 # Facturation, surconsommation, contestation montant
    WOYOFAL_RECHARGE = "WOYOFAL_RECHARGE" # Problème d'achat ou d'injection de code Woyofal
    TECHNICAL_FAULT = "TECHNICAL_FAULT" # Coupure, baisse de tension, câble/poteau endommagé
    METER_ISSUE = "METER_ISSUE"         # Compteur défectueux, affichage illisible
    SUBSCRIPTION = "SUBSCRIPTION"       # Demande de raccordement, changement de nom/police
    OTHER = "OTHER"


class ClaimScoringRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Texte de la réclamation à analyser")
    customer_id: Optional[str] = Field(None, description="Identifiant client ou numéro de police/compteur")


class ClaimScoringResponse(BaseModel):
    category: ClaimCategory = Field(..., description="Catégorie principale identifiée")
    priority: PriorityLevel = Field(..., description="Niveau de priorité attribué")
    sentiment: str = Field(..., description="Analyse du sentiment (ex: 'frustré', 'calme', 'très en colère')")
    urgency_score: float = Field(..., ge=0.0, le=1.0, description="Score d'urgence de 0.0 (faible) à 1.0 (critique)")
    summary: str = Field(..., description="Résumé en une phrase du problème signalé")
    action_recommended: str = Field(..., description="Action recommandée pour l'agent ou le système")
    model_used: str = Field(..., description="Modèle LLM ayant effectué l'analyse")
    execution_time_seconds: float = Field(..., description="Temps d'exécution en secondes")

class InternalQueryRequest(BaseModel):
    """Schéma réel utilisé par Django (ai_client.py) — distinct de
    RAGQueryRequest, qui sert une autre route (/api/v1/query, appelée
    directement par un client externe avec un vocabulaire différent)."""
    question: str = Field(..., min_length=1)
    organisation_id: str
    limit: int = Field(default=5, ge=1, le=20)
    include_live_data: bool = True
    history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Derniers échanges [{'role': 'user'|'assistant', 'content': '...'}] pour garder le fil.",
    )


class InternalQueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    model_used: str


class DocumentIndexRequest(BaseModel):
    """Correspond exactement au payload envoyé par indexer_document()
    côté Django (ai_client.py) — event_id sert à l'idempotence côté
    RagStore.index()."""
    event_id: str
    organisation_id: str
    document_id: str
    import_id: Optional[str] = None
    source: str
    texte: str = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentIndexResponse(BaseModel):
    status: str
    event_id: str
    document_id: str