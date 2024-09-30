import json
import os
import requests
import gradio as gr
import pandas as pd

class AudioWebUI:
    def __init__(self, configs):
        self.configs = configs
        self.api = "http://" + configs['rqs_host'] + ':' + str(configs['rqs_port'])
        self.speech_json = {"file_name": None, "outputs": []}  # 初始化为空的输出
        self.save_path = "tmp/"
        with gr.Row():
            with gr.Column():
                audio_path = gr.Audio(type='filepath')
                speech_button = gr.Button("1. 获取发言稿")
                save_button = gr.Button("2. 保存发言稿")
            with gr.Row():
                json_table = gr.DataFrame(value=pd.DataFrame(self.speech_json["outputs"]), label="编辑发言稿")

        speech_button.click(fn=self.get_speech, inputs=[audio_path],
                            outputs=[json_table])  # 更新表格输出
        save_button.click(fn=self.save_speech, inputs=[audio_path])  # 更新表格输出

    def get_speech(self, audio_path):
        try:
            payload = json.dumps({
                "url": audio_path,
                "batch_size": 64,
                "chunk_length_s": 30,
                "generate_kwargs": {
                    "task": "transcribe",
                    "return_timestamps": True
                }
            })
            response = requests.post(self.api+"/v1/speech/", data=payload)
            self.speech_json = response.json()
            return pd.DataFrame(self.speech_json['outputs']) 
        except Exception as e:
            print(e)
            gr.Info(f"获取发言稿失败！错误原因: {e}")
    def save_speech(self, audio_path):
        try:
            file_name = os.path.splitext(os.path.basename(audio_path))[0]
            self.speech_json['file_name'] = file_name
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            with open(f'{self.save_path}/{file_name}_speech.json', 'w', encoding='utf-8') as f:
                json.dump(self.speech_json, f, ensure_ascii=False)
            self.speech_json = {"audio_path": None, "outputs": []}
            gr.Info("保存成功！")
        except Exception as e:
            print(e)
            gr.Info(f"保存失败！错误原因: {e}")