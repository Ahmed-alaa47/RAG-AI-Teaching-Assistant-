"""
Pydantic request/response models for the AI Teaching Assistant API.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


# ─── Chat ───────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., description="The user's question (can include a YouTube URL)")
    history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Previous conversation messages for context"
    )


class SourceInfo(BaseModel):
    content: str
    metadata: Dict[str, Any]


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceInfo] = []
    presentation_path: Optional[str] = None


# ─── YouTube ────────────────────────────────────────────────────────
class YouTubeRequest(BaseModel):
    url: str = Field(..., description="Full YouTube video URL")
    question: Optional[str] = Field(
        default=None,
        description="Question to ask about the video. If provided, the AI answers from the transcript."
    )


class YouTubeResponse(BaseModel):
    video_id: str
    title: str
    duration: str
    transcript: Optional[str] = None
    answer: Optional[str] = None
    sources: List[Dict[str, Any]] = []
    error: Optional[str] = None


# ─── Recommendations ───────────────────────────────────────────────
class RecommendationRequest(BaseModel):
    topic: str = Field(..., description="Topic to search recommendations for")
    count: int = Field(default=3, ge=1, le=10)


class VideoRecommendation(BaseModel):
    title: str
    link: str
    duration: str
    type: str = "YouTube Video"


class RecommendationResponse(BaseModel):
    topic: str
    youtube: List[VideoRecommendation] = []


# ─── Presentation ──────────────────────────────────────────────────
class PresentationRequest(BaseModel):
    topic: str = Field(..., description="Topic or content for the presentation")
    title: Optional[str] = Field(default="Presentation", description="Title for the presentation")


class PresentationResponse(BaseModel):
    success: bool
    message: str
    filename: Optional[str] = None
    download_url: Optional[str] = None


# ─── Health / Status ────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    initialized: bool
    document_count: int = 0


class InitializeResponse(BaseModel):
    success: bool
    message: str
    document_count: int = 0


# ─── Upload ─────────────────────────────────────────────────────────
class UploadResponse(BaseModel):
    success: bool
    message: str
    filename: str


class DocumentAskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    filename: str


class ImageUploadResponse(BaseModel):
    success: bool
    message: str
    filename: str
    image_path: str
