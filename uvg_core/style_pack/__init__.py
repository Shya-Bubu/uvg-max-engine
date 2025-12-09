# uvg_core/style_pack/__init__.py
"""
UVG MAX Style Packs.

Provides branded export templates with:
- LUT color grading
- Transition presets
- Caption styles
- Pacing factors
- Motion curves
"""

from .base import (
    StylePack,
    load_style_pack,
    list_style_packs,
    get_default_pack,
)

__all__ = [
    "StylePack",
    "load_style_pack",
    "list_style_packs",
    "get_default_pack",
]
