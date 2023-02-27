from __future__ import annotations
import openai
import os 
from typing import List
import numpy as np
import backoff 

class ChatGPTUtil:
    
    # make function wrapper to set api key
    @staticmethod
    def set_api_key(func):
        def wrapper(*args, **kwargs):
            openai.api_key = os.getenv('OPENAI_API_KEY')
            return func(*args, **kwargs)
        return wrapper

    @classmethod
    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    @set_api_key
    def get_text_response(cls, prompt: str, engine: str = 'ada', max_tokens: int=1000, samples: int=1) -> List[str]:
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            max_tokens=max_tokens,
            n=samples)
        if 'choices' not in response:
            raise ValueError(f'No choices in response: {response}')
        return [choice['text'] for choice in response['choices']]
    
    @classmethod        
    @backoff.on_exception(backoff.constant, openai.error.RateLimitError, interval=30, jitter=None)
    @set_api_key
    def get_text_embedding(cls, input: str, model: str = 'text-embedding-ada-002') -> List[float]:
        response = openai.Embedding.create(
            model=model,
            input=input)
        if 'data' not in response:
            raise ValueError(f'No embedding in response: {response}')
        elif len(response['data']) != 1:
            raise ValueError(f'More than one embedding in response: {response}')
        elif 'embedding' not in response['data'][0]:
            raise ValueError(f'No embedding in response: {response}')
        return response['data'][0]['embedding']
       