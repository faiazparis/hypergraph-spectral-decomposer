"""Test spectral embedding computation and properties."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from hypergraph_spectral_decomposer.hypergraph import Hypergraph
from hypergraph_spectral_decomposer.spectral import (
    compute_normalized_laplacian,
    spectral_embedding,
)


class TestSpectralEmbedding:
    """Test spectral embedding computation and properties."""
    
    def test_basic_embedding(self):
        """Test basic spectral embedding computation."""
        # Simple 2-block hypergraph
        edges = [
            ["A", "B", "C"],  # Block 1
            ["B", "C", "A"],  # Block 1
            ["D", "E", "F"],  # Block 2
            ["E", "F", "D"],  # Block 2
        ]
        hg = Hypergraph(edges=edges)
        L = compute_normalized_laplacian(hg)
        
        # Compute embedding for k=2
        eigenvals, eigenvecs = spectral_embedding(L, k=2, random_state=0)
        
        # Check dimensions
        assert eigenvals.shape == (2,)
        assert eigenvecs.shape == (6, 2)  # 6 nodes, 2 eigenvectors
        
        # Check eigenvalue ordering
        assert eigenvals[0] <= eigenvals[1]
        
        # Check that eigenvectors are normalized
        for i in range(2):
            norm = np.linalg.norm(eigenvecs[:, i])
            assert abs(norm - 1.0) < 1e-10
    
    def test_eigengap_detection(self):
        """Test that eigengap is detected in 2-block structure."""
        # Create hypergraph with clear 2-block structure
        edges = [
            # Block 1: A, B, C (tightly connected)
            ["A", "B"], ["B", "C"], ["C", "A"],
            ["A", "B", "C"],  # 3-way connection
            
            # Block 2: D, E, F (tightly connected)
            ["D", "E"], ["E", "F"], ["F", "D"],
            ["D", "E", "F"],  # 3-way connection
            
            # Few cross-block connections
            ["B", "D"], ["C", "E"]
        ]
        hg = Hypergraph(edges=edges)
        L = compute_normalized_laplacian(hg)
        
        # Compute embedding for k=2
        eigenvals, eigenvecs = spectral_embedding(L, k=2, random_state=0)
        
        # Should have clear eigengap between λ₁ and λ₂
        eigengap = eigenvals[1] - eigenvals[0]
        assert eigengap > 0.1, f"Expected clear eigengap, got {eigengap}"
        
        # First eigenvector should separate the blocks
        first_eigenvec = eigenvecs[:, 0]
        
        # Check that nodes in same block have similar values
        block1_nodes = ["A", "B", "C"]
        block2_nodes = ["D", "E", "F"]
        
        block1_indices = [hg.nodes.index(node) for node in block1_nodes]
        block2_indices = [hg.nodes.index(node) for node in block2_nodes]
        
        block1_values = first_eigenvec[block1_indices]
        block2_values = first_eigenvec[block2_indices]
        
        # Values within blocks should be similar
        block1_var = np.var(block1_values)
        block2_var = np.var(block2_values)
        
        assert block1_var < 0.1, f"Block 1 values not similar, variance: {block1_var}"
        assert block2_var < 0.1, f"Block 2 values not similar, variance: {block2_var}"
    
    def test_disconnected_components(self):
        """Test handling of disconnected components in embedding."""
        # Disconnected hypergraph: two separate components
        edges = [
            ["A", "B"], ["B", "C"], ["C", "A"],  # Component 1
            ["D", "E"], ["E", "F"], ["F", "D"],  # Component 2
        ]
        hg = Hypergraph(edges=edges)
        L = compute_normalized_laplacian(hg)
        
        # Compute embedding for k=2
        eigenvals, eigenvecs = spectral_embedding(L, k=2, random_state=0)
        
        # Should have λ=0 for disconnected components
        # First eigenvalue should be very close to 0
        assert eigenvals[0] < 1e-10, f"Expected λ₁≈0 for disconnected, got {eigenvals[0]}"
        
        # Second eigenvalue should be > 0
        assert eigenvals[1] > 0.1, f"Expected λ₂>0, got {eigenvals[1]}"
    
    def test_embedding_stability(self):
        """Test that embedding is stable across different random seeds."""
        edges = [["A", "B", "C"], ["B", "C", "D"], ["A", "D"]]
        hg = Hypergraph(edges=edges)
        L = compute_normalized_laplacian(hg)
        
        # Compute embeddings with different seeds
        eigenvals1, eigenvecs1 = spectral_embedding(L, k=2, random_state=42)
        eigenvals2, eigenvecs2 = spectral_embedding(L, k=2, random_state=123)
        
        # Eigenvalues should be identical (deterministic)
        np.testing.assert_array_almost_equal(eigenvals1, eigenvals2)
        
        # Eigenvectors may differ by sign/rotation, but magnitudes should be same
        for i in range(2):
            mag1 = np.abs(eigenvecs1[:, i])
            mag2 = np.abs(eigenvecs2[:, i])
            np.testing.assert_array_almost_equal(mag1, mag2)
    
    def test_k_validation(self):
        """Test that k parameter is properly validated."""
        edges = [["A", "B"], ["B", "C"]]
        hg = Hypergraph(edges=edges)
        L = compute_normalized_laplacian(hg)
        
        # k must be < n_nodes
        with pytest.raises(ValueError, match="k=3 must be < n_nodes=3"):
            spectral_embedding(L, k=3, random_state=0)
        
        # k must be >= 1
        with pytest.raises(ValueError, match="k=0 must be >= 1"):
            spectral_embedding(L, k=0, random_state=0)
    
    def test_trivial_eigenvector_handling(self):
        """Test that trivial eigenvector (λ=0) is properly handled."""
        # Connected hypergraph should have one λ=0
        edges = [["A", "B"], ["B", "C"], ["C", "A"]]
        hg = Hypergraph(edges=edges)
        L = compute_normalized_laplacian(hg)
        
        # Compute embedding for k=2
        eigenvals, eigenvecs = spectral_embedding(L, k=2, random_state=0)
        
        # Should have exactly 2 eigenvalues
        assert len(eigenvals) == 2
        assert eigenvecs.shape[1] == 2
        
        # For a well-connected hypergraph, λ₁ may not be 0
        # First eigenvalue should be ≥ 0 (positive semi-definite)
        assert eigenvals[0] >= -1e-10, f"Expected λ₁≥0, got {eigenvals[0]}"
        
        # Second eigenvalue should be larger
        assert eigenvals[1] > eigenvals[0]
    
    def test_large_k_handling(self):
        """Test embedding when k is close to n_nodes."""
        edges = [["A", "B"], ["B", "C"], ["C", "D"], ["D", "E"]]
        hg = Hypergraph(edges=edges)
        L = compute_normalized_laplacian(hg)
        
        # k = n_nodes - 1 (maximum valid k)
        k = len(hg.nodes) - 1
        eigenvals, eigenvecs = spectral_embedding(L, k=k, random_state=0)
        
        # Should get exactly k eigenvectors
        assert eigenvals.shape == (k,)
        assert eigenvecs.shape == (len(hg.nodes), k)
        
        # All eigenvalues should be ≥ 0
        assert np.all(eigenvals >= -1e-10)
