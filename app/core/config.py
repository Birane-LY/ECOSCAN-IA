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
    # openai/gpt-oss-120b remplace llama-3.3-70b-versatile (décommissionné par
    # Groq le 16/08/2026) et mixtral-8x7b-32768 (retiré en 2025, sans
    # successeur direct) — les deux modèles précédemment configurés ici sont
    # morts, ce qui provoquait un échec total de la chaîne Groq.
    TEXT_MODEL_CHAIN_RAW: str = Field(
        default="groq:openai/gpt-oss-120b,groq:qwen/qwen3.6-27b,huggingface:meta-llama/Llama-3.3-70B-Instruct",
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

    @staticmethod
    def _nettoyer_et_parser_chaine(valeur_brute: str) -> List[str]:
        """Parse une chaîne de modèles, en tolérant qu'elle ait été saisie au
        format tableau JSON (ex: '["groq:x","groq:y"]') plutôt qu'en simple
        liste séparée par des virgules. C'est exactement l'erreur qui a cassé
        toute la chaîne de secours : le crochet et les guillemets restaient
        collés au nom du fournisseur après un simple split(','), et
        'provider == "groq"' ne matchait jamais '["groq'.

        Sans cette tolérance, une seule mauvaise ligne de .env fait échouer
        TOUTE la chaîne de secours — exactement le contraire de ce qu'un
        mécanisme de failover est censé garantir.
        """
        valeur = valeur_brute.strip()
        # Retire une éventuelle enveloppe de tableau JSON avant de split sur ','
        if valeur.startswith("[") and valeur.endswith("]"):
            valeur = valeur[1:-1]
        items = []
        for item in valeur.split(","):
            # Retire les guillemets (simples ou doubles) et espaces résiduels
            # autour de chaque élément, qu'ils viennent du format JSON ou d'une
            # saisie manuelle malheureuse.
            nettoye = item.strip().strip('"').strip("'").strip()
            if nettoye:
                items.append(nettoye)
        return items

    @property
    def TEXT_MODEL_CHAIN(self) -> List[str]:
        return self._nettoyer_et_parser_chaine(self.TEXT_MODEL_CHAIN_RAW)

    @property
    def VISION_MODEL_CHAIN(self) -> List[str]:
        return self._nettoyer_et_parser_chaine(self.VISION_MODEL_CHAIN_RAW)

    @property
    def AUDIO_TRANSCRIPTION_CHAIN(self) -> List[str]:
        return self._nettoyer_et_parser_chaine(self.AUDIO_TRANSCRIPTION_CHAIN_RAW)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()