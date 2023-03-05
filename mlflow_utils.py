import numpy as np
import json
import mlflow
from typing import Dict

def save_np_array_artifact(np_array: np.ndarray, artifact_name: str, artifact_path: str = 'experiments/artifacts'):
    np.save(f'{artifact_path}/{artifact_name}.npy', np_array)
    mlflow.log_artifact(f'{artifact_path}/{artifact_name}.npy')
    
def load_np_array_artifact(artifact_name: str, artifact_path: str = 'experiments/artifacts') -> np.ndarray:
    return np.load(f'{artifact_path}/{artifact_name}.npy')

def load_dict_artifact(artifact_name: str, artifact_path: str = 'experiments/artifacts') -> dict:
    with open(f'{artifact_path}/{artifact_name}.json', 'r') as f:
        d = json.load(f)
    return d

def log_dict_artifact(d: Dict, artifact_name: str, artifact_path: str = 'experiments/artifacts'):
    with open(f'{artifact_path}/{artifact_name}.json', 'w') as f:
        json.dump(d, f)
    mlflow.log_artifact(f'{artifact_path}/{artifact_name}.json')
    
