import os
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Gestion centralisée des paramètres de configuration de l'application EcoScan AI.
    """

    # --- Configuration de l'Application ---
    APP_NAME: str = Field(
        default="EcoScan AI/RAG",
        description="Nom principal du service FastAPI."
    )
    ENVIRONMENT: str = Field(
        default="development",
        description="Environnement d'exécution (development, staging, production)."
    )
    INTERNAL_TOKEN: str = Field(
        default="change-me-in-production",
        min_length=8,
        description="Jeton de sécurité interne pour authentifier les requêtes provenant de Django."
    )

    # --- Intégration Backend Django ---
    DJANGO_API_URL: str = Field(
        default="http://localhost:8000",
        description="URL de base de l'application Django principale."
    )
    DJANGO_INTERNAL_TOKEN: str = Field(
        default="change-me-in-production",
        description="Jeton d'authentification pour communiquer avec l'API Django."
    )
    DJANGO_TIMEOUT: float = Field(
        default=15.0,
        description="Délai d'attente maximal (en secondes) pour les requêtes HTTP vers Django."
    )

    # --- Stockage RAG et Fichiers ---
    RAG_STORE_PATH: str = Field(
        default="./data/rag_store.json",
        description="Chemin du fichier JSON servant de base de données vectorielle/textuelle pour le RAG."
    )
    MAX_UPLOAD_MB: int = Field(
        default=50,
        description="Taille maximale autorisée pour le téléversement de fichiers (en Mo)."
    )

    # --- Clés d'API Externes ---
    HUGGINGFACE_API_KEY: str = Field(
        default="",
        description="Clé d'API ou jeton d'accès pour les services d'inférence Hugging Face."
    )
    GROQ_API_KEY: str = Field(
        default="",
        description="Clé d'API Groq pour l'inférence ultra-rapide en secours."
    )
    OPENAI_API_KEY: str = Field(
        default="",
        description="Clé d'API OpenAI pour les modèles distants (optionnel)."
    )
    OPENAI_API_BASE: str = Field(
        default="",
        description="URL de base personnalisée pour les API compatibles OpenAI (ex: Groq, Ollama)."
    )

    # --- Chaînes de Secours (Failover) ---
    TEXT_MODEL_CHAIN_RAW: str = Field(
        default="groq:llama-3.3-70b-versatile,huggingface:meta-llama/Llama-3.3-70B-Instruct,huggingface:Qwen/Qwen2.5-72B-Instruct",
        alias="TEXT_MODEL_CHAIN",
        description="Liste brute séparée par des virgules des modèles texte à essayer en cascade (Groq en priorité ou backup)."
    )
    VISION_MODEL_CHAIN_RAW: str = Field(
        default="groq:llama-3.2-11b-vision-preview,huggingface:meta-llama/Llama-3.2-11B-Vision-Instruct",
        alias="VISION_MODEL_CHAIN",
        description="Liste brute séparée par des virgules des modèles vision à essayer en cascade."
    )
    AUDIO_TRANSCRIPTION_CHAIN_RAW: str = Field(
        default="local:faster-whisper,local:AIHubSN/Kiriku-Wolof-ASR",
        alias="AUDIO_TRANSCRIPTION_CHAIN",
        description="Liste brute séparée par des virgules des moteurs de transcription audio."
    )

    MIN_BACKUPS_RECOMMANDES: int = Field(
        default=2,
        description="Nombre minimal de modèles de secours recommandés pour garantir la tolérance aux pannes."
    )

    @property
    def TEXT_MODEL_CHAIN(self) -> List[str]:
        return [item.strip() for item in self.TEXT_MODEL_CHAIN_RAW.split(",") if item.strip()]

    @property
    def VISION_MODEL_CHAIN(self) -> List[str]:
        return [item.strip() for item in self.VISION_MODEL_CHAIN_RAW.split(",") if item.strip()]

    @property
    def AUDIO_TRANSCRIPTION_CHAIN(self) -> List[str]:
        return [item.strip() for item in self.AUDIO_TRANSCRIPTION_CHAIN_RAW.split(",") if item.strip()]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()