from __future__ import annotations
import openai
import os 


class ChatGPTUtil:
    
    # make function wrapper to set api key
    @staticmethod
    def set_api_key(func):
        def wrapper(*args, **kwargs):
            openai.api_key = os.getenv('OPENAI_API_KEY')
            return func(*args, **kwargs)
        return wrapper

    @classmethod
    @set_api_key
    def get_text_response(cls, prompt: str):
        return openai.Completion.create(
            engine="ada",
            prompt=prompt)