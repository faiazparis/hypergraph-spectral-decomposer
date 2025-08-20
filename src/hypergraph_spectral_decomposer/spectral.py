"""Spectral hypergraph analysis via Zhou's normalized Laplacian."""

from typing import Optional, Tuple
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

from .hypergraph import Hypergraph


def compute_normalized_laplacian(
    hypergraph: Hypergraph, weights: Optional[np.ndarray] = None, eps: float = 1e-10
) -> csr_matrix:
    """
    Compute Zhou's normalized Laplacian L = I - S.
    
    The normalized Laplacian is computed as:
        S = Dv^(-1/2) H W De^(-1) H^T Dv^(-1/2)
        L = I - S
    
    where:
        H: incidence matrix (n × m)
        Dv: node degree matrix = diag(H 1_m)
        De: edge size matrix = diag(1_n^T H)
        W: edge weight matrix (default: identity)
    
    Args:
        hypergraph: Input hypergraph
        weights: Edge weights (m-dimensional array, default: uniform)
        eps: Small value for numerical stability
        
    Returns:
        Sparse normalized Laplacian matrix L
        
    References:
        Zhou, Huang, Schölkopf (2006). Learning with Hypergraphs.
        https://papers.nips.cc/paper_files/paper/2006/file/dff8e9c2ac33381546d96deea9922999-Paper.pdf
    """
    H = hypergraph.incidence_matrix
    n_nodes, n_edges = H.shape
    
    # Default to uniform weights
    if weights is None:
        weights = np.ones(n_edges)
    
    if len(weights) != n_edges:
        raise ValueError(f"Weights length {len(weights)} != number of edges {n_edges}")
    
    # Compute degree matrices
    # Dv = diag(H 1_m) - node degrees
    node_degrees = H.sum(axis=1).A1
    node_degrees = np.maximum(node_degrees, eps)  # Numerical stability
    
    # De = diag(1_n^T H) - edge sizes  
    edge_sizes = H.sum(axis=0).A1
    edge_sizes = np.maximum(edge_sizes, eps)  # Numerical stability
    
    # Build diagonal matrices
    Dv_inv_sqrt = diags(1.0 / np.sqrt(node_degrees))
    De_inv = diags(1.0 / edge_sizes)
    W = diags(weights)
    
    # Compute S = Dv^(-1/2) H W De^(-1) H^T Dv^(-1/2)
    # This is done efficiently using sparse matrix operations
    S = Dv_inv_sqrt @ H @ W @ De_inv @ H.T @ Dv_inv_sqrt
    
    # Ensure symmetry (numerical stability)
    S = (S + S.T) / 2
    
    # Compute L = I - S
    L = diags(np.ones(n_nodes)) - S
    
    return L


def spectral_embedding(
    laplacian: csr_matrix, k: int, random_state: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute spectral embedding from normalized Laplacian.
    
    Finds the k smallest non-trivial eigenvectors of L, dropping the
    trivial constant eigenvector (λ=0) if present.
    
    Args:
        laplacian: Normalized Laplacian matrix L
        k: Number of eigenvectors to compute
        random_state: Random seed for eigendecomposition
        
    Returns:
        Tuple of (eigenvalues, eigenvectors)
        
    References:
        von Luxburg (2007). A Tutorial on Spectral Clustering.
        https://arxiv.org/abs/0711.0189
        
        Ng, Jordan, Weiss (2002). On Spectral Clustering: Analysis and an Algorithm.
        https://ai.stanford.edu/~ang/papers/nips01-spectral.pdf
    """
    n_nodes = laplacian.shape[0]
    
    if k >= n_nodes:
        raise ValueError(f"k={k} must be < n_nodes={n_nodes}")
    
    if k < 1:
        raise ValueError(f"k={k} must be >= 1")
    
    # Compute k+1 smallest eigenvalues to account for possible λ=0
    try:
        eigenvals, eigenvecs = eigsh(
            laplacian,
            k=k + 1,
            which="SM",
            random_state=random_state,
            maxiter=1000,
        )
    except Exception as e:
        # Fallback to dense computation if sparse fails
        eigenvals, eigenvecs = np.linalg.eigh(laplacian.toarray())
        eigenvals = eigenvals[:k + 1]
        eigenvecs = eigenvecs[:, :k + 1]
    
    # Sort by eigenvalue (ascending)
    sort_idx = np.argsort(eigenvals)
    eigenvals = eigenvals[sort_idx]
    eigenvecs = eigenvecs[:, sort_idx]
    
    # Check if we have λ=0 (trivial eigenvector)
    # This indicates disconnected components
    if eigenvals[0] < 1e-10:  # Numerical threshold
        # Drop the trivial eigenvector and take next k
        eigenvals = eigenvals[1:k + 1]
        eigenvecs = eigenvecs[:, 1:k + 1]
    else:
        # Take first k eigenvectors
        eigenvals = eigenvals[:k]
        eigenvecs = eigenvecs[:, :k]
    
    # Ensure we have exactly k eigenvectors
    if eigenvecs.shape[1] != k:
        # Pad with zeros if needed (shouldn't happen in practice)
        padding = np.zeros((n_nodes, k - eigenvecs.shape[1]))
        eigenvecs = np.hstack([eigenvecs, padding])
        eigenvals = np.pad(eigenvals, (0, k - len(eigenvals)), mode="constant")
    
    return eigenvals, eigenvecs


def detect_communities(
    hypergraph: Hypergraph,
    k: int,
    weights: Optional[np.ndarray] = None,
    random_state: int = 0,
    n_init: int = 10,
) -> dict:
    """
    Detect communities using spectral clustering on hypergraph.
    
    This is a convenience function that combines Laplacian computation,
    spectral embedding, and k-means clustering.
    
    Args:
        hypergraph: Input hypergraph
        k: Number of communities to detect
        weights: Optional edge weights
        random_state: Random seed for reproducibility
        n_init: Number of k-means initializations
        
    Returns:
        Dictionary mapping node_id -> community_id
        
    References:
        Chan, Louis (2018). Spectral Clustering of Graphs and Hypergraphs.
        https://arxiv.org/abs/1808.07464
    """
    # Compute normalized Laplacian
    L = compute_normalized_laplacian(hypergraph, weights)
    
    # Compute spectral embedding
    eigenvals, eigenvecs = spectral_embedding(L, k, random_state)
    
    # Apply k-means clustering
    kmeans = KMeans(
        n_clusters=k, random_state=random_state, n_init=n_init
    )
    cluster_labels = kmeans.fit_predict(eigenvecs)
    
    # Map back to node IDs
    communities = {}
    for node_idx, cluster_id in enumerate(cluster_labels):
        node_id = hypergraph.nodes[node_idx]
        communities[node_id] = int(cluster_id)
    
    return communities
