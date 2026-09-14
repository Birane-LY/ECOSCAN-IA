import logging
import time
from typing import Dict, List, Optional, Tuple
import httpx
from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMRouterService:
    """
    Service de routage des modèles de langage avec mécanisme de secours (Failover).
    
    Invoque en cascade les fournisseurs d'IA configurés (Groq, Hugging Face, OpenAI)
    jusqu'à obtenir une réponse valide.
    """

    def __init__(self):
        self.timeout = httpx.Timeout(30.0, connect=10.0)

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_chain: Optional[List[str]] = None
    ) -> Tuple[str, str]:
        """
        Exécute la génération de texte en parcourant la chaîne de secours.

        Args:
            prompt (str): Le texte ou la question de l'utilisateur.
            system_prompt (Optional[str]): Consigne système optionnelle.
            model_chain (Optional[List[str]]): Liste personnalisée de modèles à essayer.
                Si non fournie, utilise settings.TEXT_MODEL_CHAIN.

        Returns:
            Tuple[str, str]: Un tuple contenant (texte_généré, nom_du_modèle_utilisé).

        Raises:
            RuntimeError: Si tous les modèles de la chaîne de secours échouent.
        """
        chain = model_chain or settings.TEXT_MODEL_CHAIN
        errors: List[str] = []

        for model_spec in chain:
            try:
                logger.info(f"Tentative de génération avec le modèle : {model_spec}")
                provider, model_name = self._parse_model_spec(model_spec)

                if provider == "groq":
                    response_text = await self._call_groq(prompt, system_prompt, model_name)
                elif provider == "huggingface":
                    response_text = await self._call_huggingface(prompt, system_prompt, model_name)
                elif provider == "openai":
                    response_text = await self._call_openai(prompt, system_prompt, model_name)
                else:
                    raise ValueError(f"Fournisseur non pris en charge : {provider}")

                logger.info(f"Succès de la génération avec le modèle : {model_spec}")
                return response_text, model_spec

            except Exception as exc:
                error_msg = f"Échec avec {model_spec}: {str(exc)}"
                logger.warning(error_msg)
                errors.append(error_msg)

        raise RuntimeError(
            f"Tous les modèles de la chaîne de secours ont échoué. Détails : {'; '.join(errors)}"
        )

    def _parse_model_spec(self, model_spec: str) -> Tuple[str, str]:
        """
        Découpe la spécification 'fournisseur:nom_modele'.
        Exemple : 'groq:llama-3.3-70b-versatile' -> ('groq', 'llama-3.3-70b-versatile')
        """
        if ":" not in model_spec:
            # Fournisseur par défaut si non spécifié
            return "huggingface", model_spec
        provider, model_name = model_spec.split(":", 1)
        return provider.lower(), model_name

    async def _call_groq(self, prompt: str, system_prompt: Optional[str], model_name: str) -> str:
        """Appel à l'API Groq via une interface compatible OpenAI."""
        if not settings.GROQ_API_KEY:
            raise ValueError("Clé GROQ_API_KEY non configurée.")

        headers = {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.7
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def _call_huggingface(self, prompt: str, system_prompt: Optional[str], model_name: str) -> str:
        """Appel aux API d'inférence de Hugging Face."""
        if not settings.HUGGINGFACE_API_KEY:
            raise ValueError("Clé HUGGINGFACE_API_KEY non configurée.")

        headers = {
            "Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        payload = {
            "inputs": full_prompt,
            "parameters": {"max_new_tokens": 1024, "return_full_text": False}
        }
        
        url = f"https://api-inference.huggingface.co/models/{model_name}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("generated_text", "")
            elif isinstance(data, dict):
                return data.get("generated_text", data.get("content", ""))
            return str(data)

    async def _call_openai(self, prompt: str, system_prompt: Optional[str], model_name: str) -> str:
        """Appel à l'API OpenAI officielle ou à un serveur compatible."""
        if not settings.OPENAI_API_KEY:
            raise ValueError("Clé OPENAI_API_KEY non configurée.")

        base_url = settings.OPENAI_API_BASE.strip() or "https://api.openai.com/v1"
        url = f"{base_url.rstrip('/')}/chat/completions"

        headers = {
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.7
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]


# Instance singleton du routeur
llm_router = LLMRouterService()