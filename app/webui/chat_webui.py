import json
import os
import random
import gradio as gr
from openai import OpenAI
from configs.prompt import zh_prompt
class ChatWebui:
    def __init__(self,configs):
        self.configs = configs
        self.client = OpenAI(
            api_key=self.configs["api_key"],
            base_url=self.configs["base_url"],
        )
        self.prompt = zh_prompt
        self.context = []
        json_files = [f for f in os.listdir('tmp/') if f.endswith('.json')]
        json_files = gr.CheckboxGroup(choices=json_files, value=json_files[:1])
        with gr.Row():
            update_button = gr.Button("更新知识库")
            file_button = gr.Button("选择作为知识库")
        gr.ChatInterface(self.predict,submit_btn="提交",retry_btn="重试",undo_btn="撤销",clear_btn="清除")
        file_button.click(fn=self.read_speech_json,inputs=[json_files])
        update_button.click(fn=self.update_files,outputs=json_files)
    def update_files(self):
        json_files = [f for f in os.listdir('tmp/') if f.endswith('.json')]
        return gr.update(choices=json_files)
    def read_speech_json(self,json_files):
        self.context = []
        for json_file in json_files:
            with open("tmp/"+json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.context.extend(data['outputs'])
    
    def predict(self,message, history):
        history_openai_format = [{
            "role": "system",
            "content": "你是一个AI助手"
        }]
        history_openai_format.append({"role": "user", "content": self.prompt["context"].format(CONTENT=self.context)})
        for human, assistant in history:
            history_openai_format.append({"role": "user", "content": human})
            history_openai_format.append({
            "role": "assistant",
            "content": assistant
        })
        history_openai_format.append({"role": "user", "content": message})

        
        stream = self.client.chat.completions.create(
            model=self.configs["model"],
            messages=history_openai_format, 
            temperature=self.configs["temp"], 
            stream=True, 
            extra_body={
                'repetition_penalty':1
            })

        partial_message = ""
        for chunk in stream:
            partial_message += (chunk.choices[0].delta.content or "")
            yield partial_message


