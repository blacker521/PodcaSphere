# PodcaSphere
面向播客作者及播客听众的音频转文字/讲稿应用，通过音频文件生成讲稿并储存，支持说话人音色分割，使用FastAPI实现，支持异步处理，支持快速部署。
## 动机
作为播客爱好者，小宇宙的搜索流程不能满足日常的需求，不能实现对音频中的关键字进行搜索，同时没有对播客创作者提供选题或文本稿件的能力。结合whisper模型和LLM提供高质量的搜索工作流，实现高质量搜索和文本生成，后期会实现音色的克隆。

## 主要功能
- 🚀 快速部署，能够通过yaml文件快速更改配置，并部署
- 🎙️ 内置说话人音色分割功能
- 📚 底层服务为FastAPI能够通过文档接口实现所有功能，未来会提供简易webui，提供给用户使用
- 🔄 对说话人音色分割和文本处理进行了并行处理优化
- 🔗 提供异步接口进行调用
- 📃音频文件缓存和音频文件指定时段进行提取

## 使用方案

### 环境设置

`python>=3.8`

`pip install -r requirements.txt`

### 运行服务

更改`confid/deploy.yaml`文件，设置`${hf_token}`，[具体详见](https://huggingface.co/settings/tokens)，开通`pyannote/speaker-diarization-3.1`授权

运行命令

`python  main.py `

### 接口文档

访问`http://0.0.0.0:8080/docs`获取接口文档信息

使用url结尾为`async/`的接口能够提交task任务，可以通过`/status/{task_id}`和`/cancel/{task_id}`查看任务状态或取消任务，通过`/tasks`查看所有历史任务。

## ToDo

- [x] WebUI页面
- [x] 支持将文稿文件作为知识/历史记录发送给LLM模型
- [ ] 支持声纹库，通过声纹库识别说话人名称，并写入向量库
- [ ] 支持SenseVoice模型和CosyVoice模型的一键音色克隆
- [ ] 支持通过选择人物进行说话方式的Lora微调
- [ ] 支持实时语音识别
- [ ] 支持onnx模型加速

## 致谢

- [insanely-fast-whisper-api](https://github.com/JigsawStack/insanely-fast-whisper-api)
- [OpenAI's Whisper Large v3](https://huggingface.co/openai/whisper-large-v3)