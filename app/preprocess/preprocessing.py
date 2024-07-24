# _*_ coding:utf-8 _*_
"""
@date: 2024/6/19
@filename: preprocess_model_inputs
"""
import hashlib
import os

import requests
from pyannote.audio import Audio


def download_file(url: str, cache_path: str, use_cache: bool):
    '''
    :param url: audio url or audio file path
    :param cache_path: If you use HTTP/HTTPS to download, cache the files in this directory
    :param use_cache:
    :return: binary audio file
    '''
    md5_hash = hashlib.md5()
    md5_hash.update(url.encode('utf-8'))
    md5_value = md5_hash.hexdigest()
    basename, extension = os.path.splitext(url)
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    cache_path = f"{cache_path}/{md5_value}{extension}"
    if os.path.exists(cache_path) and use_cache:
        return cache_path
    else:
        inputs = requests.get(url).content
        with open(f"{cache_path}", "wb") as f:
            f.write(inputs)
        return cache_path


def prepare_audio(inputs: str, start=None, end=None):
    if inputs.startswith("http://") or inputs.startswith("https://"):
        audio_path = download_file(inputs, cache_path="../file_cache", use_cache=True)
    else:
        audio_path = inputs
    audio = Audio(sample_rate=16000)

    samples = audio(audio_path)
    return audio_path, samples[0]


def get_diarizer_inputs(audio_path: str):
    """

    :param audio_path: file path or url
    :return:audio_path,diarizer_inputs:tensor(channels, seq_len)
    """
    return prepare_audio(audio_path)


if __name__ == '__main__':
    get_diarizer_inputs("https://tk.wavpub.com/WPDL_cvHVLEGRwSAaXvAuyxFxrZudDPKSnsvDVHNwanPaXrNwByWghXyvjrgSYF-27.mp3")
