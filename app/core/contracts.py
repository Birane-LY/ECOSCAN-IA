from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


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
    description: str = Field(
        ...,
        description="Résultat textuel de l'analyse multimodale."
    )
    model_used: str = Field(
        ...,
        description="Identifiant du modèle de vision utilisé."
    )
    execution_time_seconds: float = Field(
        ...,
        description="Temps d'exécution total en secondes."
    )


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