"""Core modules for the detector web application."""

from core.model_loader import load_model
from core.web_app import create_app

__all__ = ["create_app", "load_model"]
