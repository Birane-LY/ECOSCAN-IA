import json
import os
import re
import threading
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings

# Mots vides français : sans eux, « de », « mes », « ma »... faisaient matcher
# n'importe quel document et écrasaient le classement.
MOTS_VIDES = {
    "a", "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle", "en", "et", "il", "je",
    "la", "le", "les", "leur", "lui", "ma", "mais", "me", "mes", "moi", "mon", "ne", "nos", "notre",
    "nous", "on", "ou", "par", "pas", "pour", "qu", "que", "qui", "sa", "se", "ses", "si", "son", "sur",
    "ta", "te", "tes", "toi", "ton", "tu", "un", "une", "vos", "votre", "vous", "est", "sont", "ai",
    "as", "avez", "y", "c", "d", "j", "l", "m", "n", "s", "t", "quoi", "comment", "quel", "quelle",
    "quels", "quelles", "ete", "etre", "fait", "peux", "peut", "puis",
}


def _normaliser(texte: str) -> str:
    sans_accents = unicodedata.normalize("NFKD", texte or "")
    return "".join(c for c in sans_accents if not unicodedata.combining(c)).lower()


def _tokens(texte: str) -> set:
    return {t for t in re.findall(r"\w+", _normaliser(texte)) if len(t) > 1 and t not in MOTS_VIDES}


class RagStore:
    """Index documentaire JSON, isolé par organisation. Recherche lexicale de
    secours ; remplaçable par pgvector/Qdrant sans changer l'API."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.documents: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            brut = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        cles = {"event_id", "organisation_id", "document_id", "texte", "source"}
        # Écarte les lignes corrompues (l'ancien code y écrivait un set `{...}`)
        return [d for d in brut if isinstance(d, dict) and cles <= d.keys()] if isinstance(brut, list) else []

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.documents, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def index(self, payload) -> dict:
        event_id = str(payload.event_id)
        organisation_id = str(payload.organisation_id)
        document_id = str(payload.document_id)

        with self._lock:
            deja = next((d for d in self.documents if d.get("event_id") == event_id), None)
            if deja:
                return {"status": "already_processed", "event_id": event_id, "document_id": deja["document_id"]}

            entree = {
                "event_id": event_id,
                "organisation_id": organisation_id,
                "document_id": document_id,
                "import_id": str(payload.import_id) if payload.import_id else None,
                "source": payload.source,
                "texte": payload.texte,
                "metadata": payload.metadata or {},
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }
            # Un document ré-indexé (re-validation) remplace l'ancienne version
            # au lieu de créer un doublon qui fausserait les résultats.
            self.documents = [
                d for d in self.documents
                if not (d["document_id"] == document_id and d["organisation_id"] == organisation_id)
            ]
            self.documents.append(entree)
            self._save()
        return {"status": "indexed", "event_id": event_id, "document_id": document_id}

    def delete(self, document_id: str, organisation_id: str) -> int:
        with self._lock:
            avant = len(self.documents)
            self.documents = [
                d for d in self.documents
                if not (d["document_id"] == str(document_id) and d["organisation_id"] == str(organisation_id))
            ]
            if len(self.documents) != avant:
                self._save()
            return avant - len(self.documents)

    def search(self, question: str, organisation_id: str, limit: int = 5) -> list[dict[str, Any]]:
        termes = _tokens(question)
        if not termes:
            return []
        scores = []
        for d in self.documents:
            if d["organisation_id"] != str(organisation_id):
                continue
            titre = (d.get("metadata") or {}).get("titre", "")
            mots = _tokens(f"{d['texte']} {d['source']} {titre}")
            commun = termes & mots
            if commun:
                scores.append((len(commun) / len(termes), d))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [
            {"id": d["document_id"], "source": d["source"], "text": d["texte"],
             "score": round(s, 4), "metadata": d.get("metadata", {})}
            for s, d in scores[:limit]
        ]

    def count(self) -> int:
        return len(self.documents)


rag_store = RagStore(settings.RAG_STORE_PATH)