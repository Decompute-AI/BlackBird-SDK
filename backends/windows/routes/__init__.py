"""
Routes package for Decompute Windows backend
"""

from .auth import auth_bp
from .chat import chat_bp
from .files import files_bp
from .training import training_bp
from .payment import payment_bp
from .confluence import confluence_bp

__all__ = ['auth_bp', 'chat_bp', 'files_bp', 'training_bp', 'payment_bp', 'confluence_bp']
