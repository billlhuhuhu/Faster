from typing import Callable, Dict, List


_METHOD_REGISTRY: Dict[str, Callable] = {}


def register_method(name: str):
    method_name = str(name).strip().lower()

    def decorator(func: Callable) -> Callable:
        if method_name in _METHOD_REGISTRY:
            raise ValueError(f"Baseline method '{method_name}' already registered.")
        _METHOD_REGISTRY[method_name] = func
        return func

    return decorator


def get_method(name: str) -> Callable:
    method_name = str(name).strip().lower()
    if method_name not in _METHOD_REGISTRY:
        available = ", ".join(sorted(_METHOD_REGISTRY))
        raise KeyError(f"Unknown baseline method '{method_name}'. Available: {available}")
    return _METHOD_REGISTRY[method_name]


def list_methods() -> List[str]:
    return sorted(_METHOD_REGISTRY.keys())

