# _*_ coding:utf-8 _*_
"""
@date: 2024/6/19
@filename: load_yaml
"""
import yaml


def load(fileName: str, type=None):
    with open(fileName, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as ex:
            print(ex)
    if type is not None:
        return config[type]
    else:
        return config
