from __future__ import annotations
import openai
import os 
from typing import List


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
    def get_text_response(cls, prompt: str, max_tokens: int=1000, samples: int=1) -> List[str]:
        response = openai.Completion.create(
            engine="ada",
            prompt=prompt,
            max_tokens=max_tokens,
            n=samples)
        if 'choices' not in response:
            raise ValueError('No choices in response')
        return [choice['text'] for choice in response['choices']]