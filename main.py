import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import certifi
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pymongo import MongoClient
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError


app = FastAPI(title="ISTQB Backend API", version="1.0.0")


def _get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@lru_cache(maxsize=1)
def get_mongo_client() -> MongoClient:
    mongo_uri = _get_env("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MONGO_URI is required.")
    mongo_kwargs: Dict[str, Any] = {
        "serverSelectionTimeoutMS": int(_get_env("MONGO_SERVER_SELECTION_TIMEOUT_MS", "12000")),
        "connectTimeoutMS": int(_get_env("MONGO_CONNECT_TIMEOUT_MS", "12000")),
        "socketTimeoutMS": int(_get_env("MONGO_SOCKET_TIMEOUT_MS", "20000")),
        "tlsCAFile": certifi.where(),
        "retryWrites": True,
    }
    client = MongoClient(mongo_uri, **mongo_kwargs)
    client.admin.command("ping")
    return client


def get_collections():
    db_name = _get_env("MONGO_DB_NAME", "ISTQBExam")
    sets_collection = _get_env("MONGO_SETS_COLLECTION", "sets")
    questions_collection = _get_env("MONGO_QUESTIONS_COLLECTION", "questions")
    db = get_mongo_client()[db_name]
    return db[sets_collection], db[questions_collection]


def _extract_bearer(authorization: Optional[str]) -> str:
    if not authorization:
        return ""
    parts = authorization.strip().split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return ""
    return parts[1].strip()


def auth_guard(
    authorization: Optional[str] = Header(default=None),
    x_api_token: Optional[str] = Header(default=None, alias="X-API-Token"),
) -> None:
    expected = _get_env("API_AUTH_TOKEN")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server misconfigured: missing API_AUTH_TOKEN",
        )

    provided = _extract_bearer(authorization) or (x_api_token or "").strip()
    if provided != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )


def _clean_set_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(doc.get("_id")),
        "set_number": doc.get("set_number"),
        "code": doc.get("code"),
        "name": doc.get("name"),
        "order": doc.get("order"),
        "is_active": doc.get("is_active", True),
        "question_count": doc.get("question_count"),
    }


def _normalize_correct_indices(doc: Dict[str, Any]) -> List[int]:
    if isinstance(doc.get("correct_indices"), list):
        return [int(v) for v in doc["correct_indices"] if isinstance(v, int)]
    if isinstance(doc.get("correct_index"), int):
        return [doc["correct_index"]]
    return []


def _clean_question_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(doc.get("_id")),
        "set_id": str(doc.get("set_id")) if doc.get("set_id") is not None else None,
        "set_number": doc.get("set_number"),
        "set_code": doc.get("set_code"),
        "order": doc.get("order"),
        "question": doc.get("question"),
        "options": doc.get("options", []),
        "correct_indices": _normalize_correct_indices(doc),
        "is_active": doc.get("is_active", True),
        "tag": doc.get("tag"),
        "learning_objective": doc.get("learning_objective"),
        "k_level": doc.get("k_level"),
        "points": doc.get("points"),
        "explanation": doc.get("explanation"),
    }


@app.head("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/sets", dependencies=[Depends(auth_guard)])
def get_sets() -> Dict[str, List[Dict[str, Any]]]:
    try:
        sets_col, _ = get_collections()
    except ServerSelectionTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection timeout (TLS/network/IP allowlist).",
        ) from exc
    except (RuntimeError, PyMongoError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database unavailable: {exc.__class__.__name__}",
        ) from exc

    docs = list(
        sets_col.find({"is_active": {"$ne": False}}).sort([("order", 1), ("set_number", 1)])
    )
    return {"items": [_clean_set_doc(doc) for doc in docs]}


@app.get("/sets/{set_number}/questions", dependencies=[Depends(auth_guard)])
def get_questions_by_set(set_number: int) -> Dict[str, List[Dict[str, Any]]]:
    try:
        _, questions_col = get_collections()
    except ServerSelectionTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection timeout (TLS/network/IP allowlist).",
        ) from exc
    except (RuntimeError, PyMongoError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database unavailable: {exc.__class__.__name__}",
        ) from exc

    docs = list(
        questions_col.find(
            {"set_number": set_number, "is_active": {"$ne": False}},
        ).sort([("order", 1), ("_id", 1)])
    )
    return {"items": [_clean_question_doc(doc) for doc in docs]}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

