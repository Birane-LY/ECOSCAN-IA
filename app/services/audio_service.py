import logging
import time
import httpx
from typing import Tuple, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

class AudioService:
    """Service de transcription audio s'appuyant sur Whisper via Groq API."""

    def __init__(self):
        self.timeout = httpx.Timeout(60.0, connect=10.0)

    async def transcribe_audio(
        self, 
        file_bytes: bytes, 
        filename: str, 
        language: Optional[str] = None
    ) -> Tuple[str, Optional[str], str, float]:
        """
        Transcrit un fichier audio en texte.

        Returns:
            Tuple[str, Optional[str], str, float]: 
            (transcription, language_detected, engine_used, execution_time_seconds)
        """
        if not settings.GROQ_API_KEY:
            raise ValueError("Clé GROQ_API_KEY non configurée pour la transcription audio.")

        start_time = time.time()
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}"
        }

        files = {
            "file": (filename, file_bytes, "audio/mpeg")
        }
        data = {
            "model": "whisper-large-v3",
            "response_format": "json"
        }
        if language:
            data["language"] = language

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, headers=headers, files=files, data=data)
            resp.raise_for_status()
            result = resp.json()
            
            execution_time = round(time.time() - start_time, 3)
            transcription = result.get("text", "")
            detected_lang = result.get("language", language)

            return transcription, detected_lang, "groq:whisper-large-v3", execution_time

audio_service = AudioService()