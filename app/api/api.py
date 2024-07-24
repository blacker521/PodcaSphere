# _*_ coding:utf-8 _*_
"""
@date: 2024/6/18
@filename: api
"""
import asyncio
import uuid

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException

from app.api.models import *
from app.postprocess.postprocessing import diarize_audio, post_process_segments_and_transcripts
from app.preprocess.preprocessing import get_diarizer_inputs
from app.service.model_service import WhisperService, DiarizeService


class Api:
    def __init__(self, app: FastAPI, config):
        self.router = APIRouter()
        self.app = app
        self.config = config
        self.add_api_route("/v1/speech/", self.speech_outputs, methods=["POST"],
                           response_model=SpeechResponse)
        self.add_api_route("/v1/whisper/", self.whisper_outputs, methods=["POST"],
                           response_model=WhisperResponse)
        self.add_api_route("/v1/diarize/", self.diarizer_outputs, methods=["POST"],
                           response_model=DiarizerResponse)
        self.add_api_route("/v1/speech/async/", self.async_audio2speech, methods=["POST"])
        self.add_api_route("/v1/whisper/async/", self.async_whisper_outputs, methods=["POST"])
        self.add_api_route("/v1/diarize/async/", self.async_diarizer_outputs, methods=["POST"])
        self.add_api_route("/cancel/{task_id}", self.cancel_task, methods=["DELETE"])
        self.add_api_route("/status/{task_id}", self.task_status, methods=["GET"])
        self.add_api_route("/tasks", self.tasks, methods=["GET"])
        self.whisper_service = WhisperService(self.config["whisper"])
        self.diarize_service = DiarizeService(self.config["diarize"])
        self.running_tasks = {}

    def add_api_route(self, path: str, endpoint, **kwargs):
        return self.app.add_api_route(path, endpoint, **kwargs)

    def tasks(self):
        return TasksResponse(tasks=list(self.running_tasks.keys()))

    def task_status(self, task_id: str):
        if task_id not in self.running_tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        task = self.running_tasks[task_id]
        if task is None:
            return TaskStatuResponse(status="processing")
        elif task.done() is False:
            return TaskStatuResponse(status="processing")
        else:
            return TaskStatuResponse(status="completed", outputs=task.result().outputs)

    def cancel_task(self, task_id: str):
        if task_id not in self.running_tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        task = self.running_tasks[task_id]
        if task is None:
            return HTTPException(status_code=400, detail="Not a background task")
        elif task.done() is False:
            task.cancel()
            del self.running_tasks[task_id]
            return TaskStatuResponse(status="cancelled")
        else:
            return TaskStatuResponse(status="completed", outputs=task.result().outputs)

    def launch(self):
        self.app.include_router(self.router)
        uvicorn.run(
            self.app,
            host=self.config["api"]["host"],
            port=self.config["api"]["port"],
            timeout_keep_alive=self.config["api"]["timeout_keep_alive"],
            root_path=""
        )

    async def whisper_outputs(self, req: WhisperRequet):
        audio_file_path, _ = get_diarizer_inputs(req.url)
        outputs = await self.whisper_service(audio_file_path, req.batch_size, req.chunk_length_s,
                                             req.generate_kwargs)
        return WhisperResponse(text=outputs['text'],
                               outputs=[WhisperChunk(text=chunk['text'], start=chunk["timestamp"][0],
                                                     end=chunk["timestamp"][1]) for chunk in
                                        outputs["chunks"]])

    async def async_audio2speech(self, req: SpeechRequest):
        return self.async_background_task(self.speech_outputs, req)

    async def async_diarizer_outputs(self, req: DiarizerRequest):
        return self.async_background_task(self.diarizer_outputs, req)

    async def async_whisper_outputs(self, req: WhisperRequet):
        return self.async_background_task(self.whisper_outputs, req)

    def async_background_task(self, fun, req):
        task_id = str(uuid.uuid4())
        loop = asyncio.get_event_loop()
        task = loop.create_task(fun(req))
        self.running_tasks[task_id] = task
        return CreateTaskResponse(status="processing", task_id=task_id)

    async def speech_outputs(self, req: SpeechRequest):
        audio_file_path, diarize_inputs = get_diarizer_inputs(req.url)
        loop = asyncio.get_event_loop()
        whisper_task = loop.create_task(self.whisper_service(audio_file_path, req.batch_size, req.chunk_length_s,
                                                             req.generate_kwargs))
        segments = await self.diarize_service(diarize_inputs, {"sample_rate": 16000})
        new_segments = diarize_audio(segments)
        whisper_outputs = await whisper_task
        return SpeechResponse(
            outputs=post_process_segments_and_transcripts(new_segments, whisper_outputs['chunks'],
                                                          group_by_speaker=True))

    async def diarizer_outputs(self, req: DiarizerRequest):
        audio_file_path, diarize_inputs = get_diarizer_inputs(req.url)
        diarize_outputs = await self.diarize_service(diarize_inputs, req.generate_kwargs)
        segments = diarize_audio(diarize_outputs)
        return DiarizerResponse(outputs=segments)
