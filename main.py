# _*_ coding:utf-8 _*_
"""
@date: 2024/6/18
@filename: webui
"""
import threading
from app.uitls.load_yaml import load
import gradio as gr

from app.webui.audio_webui import AudioWebUI
from app.webui.chat_webui import ChatWebui
CONFIG_PATH = "configs/deploy.yaml"
config = load(CONFIG_PATH)


def create_api(app):
    from app.api.api import Api
    api = Api(app, config)
    return api


def api_only():
    from fastapi import FastAPI
    app = FastAPI()
    api = create_api(app)
    api.launch()


def webui():
        api_thread = threading.Thread(target=api_only)
    api_thread.daemon = True
    api_thread.start()
    interface = gr.Blocks(title='PodcaSphere')
    with interface:
        with gr.Tab('PodcaSphere'):
            AudioWebUI(config["webui"])
        with gr.Tab('Chat'):
            ChatWebui(config["openai"])
    interface.launch(inbrowser=True, share=True, server_name=config["webui"]["host"], server_port=config["webui"]["port"])



if __name__ == "__main__":

    webui()
