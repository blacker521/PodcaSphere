# _*_ coding:utf-8 _*_
"""
@date: 2024/6/20
@filename: model_service
"""
import torch
from pyannote.audio import Pipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class DiarizeService:
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device(config['device'])
        self.pipeline = Pipeline.from_pretrained(
            checkpoint_path=config["checkpoint_path"],
            use_auth_token=config['hf_token'],
        )
        self.pipeline.to(self.device)

    async def __call__(self, torch_inputs, generate_kwargs):
        torch_inputs = torch_inputs.to(self.device)
        outputs = self.pipeline(
            {"waveform": torch_inputs, **generate_kwargs},
        )
        return outputs


class WhisperService:
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device(config['device'])
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            config["checkpoint_path"], torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(config["checkpoint_path"], )
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=self.device,
        )

    async def __call__(self, url, batch_size, chunk_length_s, generate_kwargs):
        outputs = self.pipe(
            url,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            generate_kwargs=generate_kwargs,
            return_timestamps=True
        )
        return outputs
