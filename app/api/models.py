# _*_ coding:utf-8 _*_
"""
@date: 2024/6/18
@filename: models
"""
from typing import List, Union

from pydantic import BaseModel, Field


class SegmentSpeakerResponse(BaseModel):
    start: float = Field(default=None, title="id", description="")
    end: float = Field(default=None, title="id", description="")
    speaker: str = Field(default=None, title="id", description="")
    text: str = Field(default=None, title="id", description="")


class SpeakerChunk(BaseModel):
    speaker: str = Field(default=None, title="speaker", description="")
    end: float = Field(default=None, title="end", description="")
    start: float = Field(default=None, title="strat", description="")


class DiarizerRequest(BaseModel):
    url: str
    generate_kwargs: dict = Field(default={"sample_rate": 16000}, title="generate_kwargs", description="")


class DiarizerResponse(BaseModel):
    status: str = Field(default="completed", title="status", description="")
    outputs: List[SpeakerChunk]


class WhisperRequet(BaseModel):
    url: str
    batch_size: int = Field(default=64, title="speaker", description="")
    chunk_length_s: int = Field(default=30, title="speaker", description="")
    generate_kwargs: dict = Field(default={
        "task": "transcribe",
        "language": "chinese",
        "return_timestamps": True,
    }, title="generate_kwargs", description="")


class WhisperChunk(BaseModel):
    text: str
    start: float
    end: float


class WhisperResponse(BaseModel):
    status: str = Field(default="completed", title="status", description="")
    outputs: List[WhisperChunk]


class SpeechRequest(BaseModel):
    url: str = Field(title="url", description="audio path or oss url")
    batch_size: int = Field(default=64, title="speaker", description="")
    chunk_length_s: int = Field(default=30, title="speaker", description="")
    generate_kwargs: dict = Field(default={
        "task": "transcribe",
        "language": "chinese",
        "return_timestamps": True,
    }, title="generate_kwargs", description="")


class SpeechResponse(BaseModel):
    status: str = Field(default="completed", title="status", description="")
    outputs: List[SegmentSpeakerResponse] = Field(default=None, title="outputs", description="")


class TaskStatuResponse(BaseModel):
    status: str
    outputs: Union[str, List[WhisperChunk], List[SpeakerChunk], List[SegmentSpeakerResponse]]


class TasksResponse(BaseModel):
    tasks: List[str]


class CreateTaskResponse(BaseModel):
    status: str
    task_id: str
    detail: str = Field(default="Run task in background")
