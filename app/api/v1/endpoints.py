import time
import mimetypes
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status

from app.core.contracts import (
    RAGQueryRequest, RAGQueryResponse,
    InternalQueryRequest, InternalQueryResponse,
    DocumentIndexRequest, DocumentIndexResponse,
    AudioTranscriptionResponse,
    VisionAnalysisResponse,
    ClaimScoringRequest, ClaimScoringResponse,
)
from app.core.security import verify_internal_token
from app.services.audio_service import audio_service
from app.services.vision_service import vision_service, VisionExtractionError
from app.services.rag_store import rag_store
from app.services import assistant_service
from app.services.scoring_service import scoring_service, DEFAULT_SCORING_MODEL
from app.services.llm_router import llm_router

router = APIRouter()

MODELE_VISION_PAR_DEFAUT = "qwen/qwen3.6-27b"
EXTENSIONS_IMAGE_AUTORISEES = (".jpg", ".jpeg", ".png", ".webp")
ALLOWED_AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac")

@router.post(
    "/internal/query",
    response_model=InternalQueryResponse,
    dependencies=[Depends(verify_internal_token)],
    summary="Assistant conversationnel EcoScan (RAG + données live)",
)
async def internal_query(payload: InternalQueryRequest) -> InternalQueryResponse:
    resultat = await assistant_service.repondre(
        question=payload.question,
        organisation_id=payload.organisation_id,
        limit=payload.limit,
        include_live_data=payload.include_live_data,
        history=payload.history,
    )
    return InternalQueryResponse(**resultat)


@router.post(
    "/internal/documents",
    response_model=DocumentIndexResponse,
    dependencies=[Depends(verify_internal_token)],
    summary="Indexe un contenu déjà validé dans le RAG",
)
async def internal_documents(payload: DocumentIndexRequest) -> DocumentIndexResponse:
    resultat = rag_store.index(payload)
    return DocumentIndexResponse(**resultat)


# --- Endpoints publics v1 (montés sous /api/v1 dans main.py) ---

@router.post(
    "/query",
    response_model=RAGQueryResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_internal_token)],
    summary="Exécuter une requête texte via le routeur LLM (sans RAG documentaire)",
)
async def query_llm(request: RAGQueryRequest) -> RAGQueryResponse:
    start_time = time.time()
    try:
        system_prompt = (
            "Tu es un assistant IA expert intégré au service EcoScan. "
            "Réponds de manière claire, concise, précise et structurée."
        )
        answer_text, model_used = await llm_router.generate_text(prompt=request.query, system_prompt=system_prompt)
        return RAGQueryResponse(
            answer=answer_text, sources=[], model_used=model_used,
            execution_time_seconds=round(time.time() - start_time, 3),
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Erreur lors du traitement : {exc}")


@router.post(
    "/audio/transcribe",
    response_model=AudioTranscriptionResponse,
    dependencies=[Depends(verify_internal_token)],
    summary="Transcrit un fichier audio en texte",
)
async def transcribe_audio_endpoint(file: UploadFile = File(...), language: Optional[str] = Form(default=None)):
    if not file.filename.lower().endswith(ALLOWED_AUDIO_EXTENSIONS):
        raise HTTPException(status_code=400, detail=f"Format non pris en charge. Formats acceptés : {', '.join(ALLOWED_AUDIO_EXTENSIONS)}")
    try:
        content = await file.read()
        transcription, detected_lang, engine, execution_time = await audio_service.transcribe_audio(
            file_bytes=content, filename=file.filename, language=language,
        )
        return AudioTranscriptionResponse(
            transcription=transcription, language_detected=detected_lang,
            engine_used=engine, execution_time_seconds=execution_time,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la transcription audio : {exc}")


@router.post(
    "/image/analyze",
    response_model=VisionAnalysisResponse,
    dependencies=[Depends(verify_internal_token)],
    summary="Extrait les données d'une facture Senelec ou d'un compteur électrique",
)
async def analyze_image_endpoint(file: UploadFile = File(...), modele: Optional[str] = Form(default=None)) -> VisionAnalysisResponse:
    filename_lower = (file.filename or "").lower()
    if not filename_lower.endswith(EXTENSIONS_IMAGE_AUTORISEES):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Format d'image non pris en charge. Formats acceptés : {', '.join(EXTENSIONS_IMAGE_AUTORISEES)}")

    mime_type, _ = mimetypes.guess_type(filename_lower)
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Le fichier image est vide.")

        champs_extraits, champs_non_lisibles, execution_time = await vision_service.analyser_facture_ou_compteur(
            image_bytes=content, mime_type=mime_type, modele=modele,
        )
        model_used = modele or MODELE_VISION_PAR_DEFAUT
        champs_detectes = [f"{cle}: {valeur}" for cle, valeur in champs_extraits.items() if valeur is not None]
        description = (
            "Extraction réussie. Champs détectés : " + "; ".join(champs_detectes) + "."
            if champs_detectes else "Aucune donnée clairement lisible n'a été détectée sur l'image."
        )
        return VisionAnalysisResponse(
            description=description, champs=champs_extraits, champs_non_lisibles=champs_non_lisibles,
            model_used=model_used, execution_time_seconds=execution_time,
        )
    except HTTPException:
        raise
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err)) from val_err
    except VisionExtractionError as ext_err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Échec de l'extraction : {ext_err}") from ext_err
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Erreur lors du traitement de l'image : {exc}") from exc
    finally:
        await file.close()


@router.post(
    "/claims/analyze",
    response_model=ClaimScoringResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_internal_token)],
    summary="Analyse le sentiment, la catégorie et la priorité d'une réclamation",
)
async def analyze_claim_endpoint(payload: ClaimScoringRequest):
    try:
        parsed, execution_time = await scoring_service.analyze_claim(payload.text)
        cles = ("category", "priority", "sentiment", "urgency_score", "summary", "action_recommended")
        return ClaimScoringResponse(
            **{k: parsed[k] for k in cles if k in parsed},
            model_used=f"groq:{DEFAULT_SCORING_MODEL}",
            execution_time_seconds=execution_time,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))