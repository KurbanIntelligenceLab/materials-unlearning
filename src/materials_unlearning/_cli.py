"""Compatibility shim for legacy imports."""

from .cli.shared import parser, load, make_backend, split  # re-export shared CLI helpers

__all__ = ["parser", "load", "make_backend", "split"]
