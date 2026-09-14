import time
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from app.core.contracts import RAGQueryRequest, RAGQueryResponse,  AudioTranscriptionResponse
from app.core.security import verify_internal_token
from app.services.llm_router import llm_router
from typing import Optional
from app.services.audio_service import audio_service
import mimetypes
from app.core.contracts import VisionAnalysisResponse
from app.services.vision_service import vision_service, VisionExtractionError



router = APIRouter()


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Exécuter une requête texte ou RAG via le routeur LLM",
    description="Reçoit une question texte, invoque la chaîne de secours des LLM et retourne la réponse générée."
)
async def query_llm(
    request: RAGQueryRequest,
    _: str = Depends(verify_internal_token)
) -> RAGQueryResponse:
    """
    Endpoint principal pour les requêtes texte / RAG.
    
    Exécute le routeur LLM asynchrone avec gestion de la tolérance aux pannes.
    """
    start_time = time.time()

    try:
        # Prompt système de base pour le service EcoScan
        system_prompt = (
            "Tu es un assistant IA expert intégré au service EcoScan. "
            "Réponds de manière claire, concise, précise et structurée."
        )

        # Appel du routeur LLM (failover automatique)
        answer_text, model_used = await llm_router.generate_text(
            prompt=request.query,
            system_prompt=system_prompt
        )

        execution_time = round(time.time() - start_time, 3)

        return RAGQueryResponse(
            answer=answer_text,
            sources=[],  # RAG vectoriel à intégrer dans la feature dédiée
            model_used=model_used,
            execution_time_seconds=execution_time
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du traitement de la requête LLM : {str(exc)}"
        )


router = APIRouter()

@router.post(
    "/audio/transcribe",
    response_model=AudioTranscriptionResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_internal_token)],
    summary="Transcrit un fichier audio de réclamation en texte"
)
async def transcribe_audio_endpoint(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None)
):
    """
    Accepte un fichier audio (MP3, WAV, M4A, OGG) et retourne la transcription textuelle.
    """
    allowed_extensions = (".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac")
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Format non pris en charge. Formats acceptés : {', '.join(allowed_extensions)}"
        )

    try:
        content = await file.read()
        transcription, detected_lang, engine, execution_time = await audio_service.transcribe_audio(
            file_bytes=content,
            filename=file.filename,
            language=language
        )

        return AudioTranscriptionResponse(
            transcription=transcription,
            language_detected=detected_lang,
            engine_used=engine,
            execution_time_seconds=execution_time
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la transcription audio : {str(exc)}"
        )


router = APIRouter()

MODELE_VISION_PAR_DEFAUT = "qwen/qwen3.6-27b"
EXTENSIONS_IMAGE_AUTORISEES = (".jpg", ".jpeg", ".png", ".webp")


@router.post(
    "/image/analyze",
    response_model=VisionAnalysisResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_internal_token)],
    summary="Extrait les données d'une facture Senelec ou d'un compteur électrique",
)
async def analyze_image_endpoint(
    file: UploadFile = File(...),
    modele: Optional[str] = Form(
        default=None,
        description="Modèle vision optionnel pour surcharger la valeur par défaut",
    ),
) -> VisionAnalysisResponse:
    """
    Analyse un document énergétique ou une photo de compteur et extrait les
    champs structurés : numéro de facture, police, compteur, consommation et montant.
    """

    filename = file.filename or ""
    filename_lower = filename.lower()

    if not filename_lower.endswith(EXTENSIONS_IMAGE_AUTORISEES):
        formats_acceptes = ", ".join(EXTENSIONS_IMAGE_AUTORISEES)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Format d'image non pris en charge. "
                f"Formats acceptés : {formats_acceptes}"
            ),
        )

    mime_type, _ = mimetypes.guess_type(filename_lower)
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"

    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Le fichier image est vide.",
            )

        (
            champs_extraits,
            champs_non_lisibles,
            execution_time,
        ) = await vision_service.analyser_facture_ou_compteur(
            image_bytes=content,
            mime_type=mime_type,
            modele=modele,
        )

        model_used = modele or MODELE_VISION_PAR_DEFAUT
        champs_detectes = [
            f"{cle}: {valeur}"
            for cle, valeur in champs_extraits.items()
            if valeur is not None
        ]

        if champs_detectes:
            description = (
                "Extraction réussie. Champs détectés : "
                + "; ".join(champs_detectes)
                + "."
            )
        else:
            description = (
                "Aucune donnée clairement lisible n'a été détectée sur l'image."
            )

        return VisionAnalysisResponse(
            description=description,
            champs=champs_extraits,
            champs_non_lisibles=champs_non_lisibles,
            model_used=model_used,
            execution_time_seconds=execution_time,
        )

    except HTTPException:
        raise

    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        ) from val_err

    except VisionExtractionError as ext_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Échec de l'extraction de données de l'image : "
                f"{ext_err}"
            ),
        ) from ext_err

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du traitement de l'image : {exc}",
        ) from exc

    finally:
        await file.close()
