from __future__ import annotations

import threading
from typing import Any, Dict

_model_cache: Dict[str, Any] = {}
_cache_lock = threading.Lock()


def load_sentence_transformer(model_name: str, **kwargs: Any):
    global _model_cache
    with _cache_lock:
        if model_name in _model_cache:
            return _model_cache[model_name]

    from sentence_transformers import SentenceTransformer

    model_kwargs = dict(kwargs.pop("model_kwargs", None) or {})
    model_kwargs.setdefault("low_cpu_mem_usage", False)
    
    model = SentenceTransformer(model_name, model_kwargs=model_kwargs, **kwargs)
    
    with _cache_lock:
        _model_cache[model_name] = model
        
    return model
