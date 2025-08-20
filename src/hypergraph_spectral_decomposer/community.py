"""Community detection interface for hypergraphs."""

from typing import Dict, Optional
import numpy as np

from .hypergraph import Hypergraph
from .spectral import detect_communities as _spectral_detect


def detect_communities(
    hypergraph: Hypergraph,
    k: int,
    weights: Optional[np.ndarray] = None,
    random_state: int = 0,
    n_init: int = 10,
) -> Dict[str, int]:
    """
    Detect communities in hypergraph using spectral clustering.
    
    This is the main entry point for community detection. It implements
    the complete pipeline: normalized Laplacian computation, spectral
    embedding, and k-means clustering.
    
    Args:
        hypergraph: Input hypergraph
        k: Number of communities to detect
        weights: Optional edge weights (m-dimensional array)
        random_state: Random seed for reproducibility
        n_init: Number of k-means initializations
        
    Returns:
        Dictionary mapping node_id -> community_id (0 to k-1)
        
    Raises:
        ValueError: If k is invalid or hypergraph is empty
        
    References:
        Zhou, Huang, Schölkopf (2006). Learning with Hypergraphs.
        https://papers.nips.cc/paper_files/paper/2006/file/dff8e9c2ac33381546d96deea9922999-Paper.pdf
        
        von Luxburg (2007). A Tutorial on Spectral Clustering.
        https://arxiv.org/abs/0711.0189
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    
    if len(hypergraph) < k:
        raise ValueError(f"Number of nodes ({len(hypergraph)}) must be >= k ({k})")
    
    if len(hypergraph.edges) == 0:
        raise ValueError("Hypergraph has no edges")
    
    # Validate weights if provided
    if weights is not None:
        if len(weights) != len(hypergraph.edges):
            raise ValueError(
                f"Weights length ({len(weights)}) must match number of edges ({len(hypergraph.edges)})"
            )
        if np.any(weights < 0):
            raise ValueError("All weights must be non-negative")
    
    # Delegate to spectral implementation
    return _spectral_detect(
        hypergraph=hypergraph,
        k=k,
        weights=weights,
        random_state=random_state,
        n_init=n_init,
    )
