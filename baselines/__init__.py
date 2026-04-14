"""Independent multimodal baseline selection framework."""

from .registry import get_method, list_methods, register_method

__all__ = ["get_method", "list_methods", "register_method"]

