"""Hypergraph Spectral Decomposer - Community detection via Zhou's normalized Laplacian."""

__version__ = "0.1.0"

from .hypergraph import Hypergraph
from .spectral import spectral_embedding
from .community import detect_communities

__all__ = ["Hypergraph", "spectral_embedding", "detect_communities"]
