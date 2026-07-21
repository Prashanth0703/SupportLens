from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check(request: Request) -> dict[str, str | bool]:
    model_loaded = (
        getattr(request.app.state, "intent_predictor", None) is not None
    )
    return {
        "status": "ok",
        "service": "supportlens-api",
        "model_loaded": model_loaded,
    }
