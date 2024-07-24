# PodcaSphere

An audio-to-text/speech transcription application for podcast authors and listeners, which generates transcripts from audio files and stores them, supports speaker voice tone segmentation, implemented with FastAPI, supports asynchronous processing, and easy deployment.

## Motivation

As a podcast enthusiast, the search process on Xiaoyuzhou does not meet daily needs, unable to search for keywords in audio, and does not provide podcast creators with topics or text manuscripts. Combining the whisper model and LLM to provide high-quality search workflows, achieve high-quality searches and text generation, and later realize voice cloning.

## Main Features

- 🚀 Easy Deployment: Quickly change configurations and deploy through a yaml file.
- 🎙️ Built-in Speaker Voice Tone Segmentation Functionality.
- 📚 Underlying service is FastAPI, all features can be accessed through documentation interfaces, a simple web UI will be provided in the future for user use.
- 🔄 Parallel processing optimization for speaker voice tone segmentation and text processing.
- 🔗 Provides asynchronous interfaces for calling.
- 📃 Audio file caching and extraction of specified time periods from audio files.

## Usage Plan

### Environment Setup

```
python>=3.8
pip install -r requirements.txt
```

### Running the Service

Change the `confid/deploy.yaml` file, set `${hf_token}`, [see details here](https://huggingface.co/settings/tokens), and authorize `pyannote/speaker-diarization-3.1`.

Run the command 

`python  main.py `

### API Documentation

Visit `http://0.0.0.0:8080/docs` to get the API documentation information.

Using the endpoint that ends with `async/` allows you to submit task tasks, you can check the task status or cancel the task through `/status/{task_id}` and `/cancel/{task_id}`, and view all historical tasks through `/tasks`.

## To Do

- [ ] Web UI page
- [ ] Support sending script files as knowledge/history records to the LLM model.
- [ ] Support voiceprint library, identify speaker names through voiceprint library and write them into the vector library.
- [ ] Support one-click voice cloning with SenseVoice model and CosyVoice model.
- [ ] Support Lora fine-tuning for speaking style by selecting characters.
- [ ] Support real-time speech recognition.
- [ ] Support ONNX model acceleration.

## Acknowledgements

- [insanely-fast-whisper-api](https://github.com/JigsawStack/insanely-fast-whisper-api)
- [OpenAI's Whisper Large v3](https://huggingface.co/openai/whisper-large-v3)