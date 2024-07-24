# _*_ coding:utf-8 _*_
"""
@date: 2024/6/18
@filename: webui
"""
from app.uitls.load_yaml import load

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
    pass


if __name__ == "__main__":
    api_only()
