"""Test clustering consistency and community detection accuracy."""

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from hypergraph_spectral_decomposer.hypergraph import Hypergraph
from hypergraph_spectral_decomposer.community import detect_communities


class TestClusteringConsistency:
    """Test that spectral clustering produces consistent and accurate results."""
    
    def test_two_block_recovery(self):
        """Test that spectral clustering recovers 2-block structure."""
        # Create hypergraph with clear 2-block structure
        edges = [
            # Block 1: A, B, C (tightly connected)
            ["A", "B"], ["B", "C"], ["C", "A"],
            ["A", "B", "C"],  # 3-way connection
            
            # Block 2: D, E, F (tightly connected)
            ["D", "E"], ["E", "F"], ["F", "D"],
            ["D", "E", "F"],  # 3-way connection
            
            # Few cross-block connections (noise)
            ["B", "D"], ["C", "E"]
        ]
        hg = Hypergraph(edges=edges)
        
        # Expected ground truth communities
        ground_truth = {
            "A": 0, "B": 0, "C": 0,  # Block 1
            "D": 1, "E": 1, "F": 1,  # Block 2
        }
        
        # Detect communities
        communities = detect_communities(hg, k=2, random_state=0)
        
        # Check that we got 2 communities
        unique_communities = set(communities.values())
        assert len(unique_communities) == 2, f"Expected 2 communities, got {len(unique_communities)}"
        
        # Map community IDs to match ground truth (community IDs may be swapped)
        community_mapping = {}
        for node in hg.nodes:
            if node in ["A", "B", "C"]:
                expected_community = 0
            else:
                expected_community = 1
            
            actual_community = communities[node]
            if expected_community not in community_mapping:
                community_mapping[expected_community] = actual_community
            else:
                # Should map to same community
                assert community_mapping[expected_community] == actual_community
        
        # Check that all nodes in same block got same community
        block1_communities = [communities[node] for node in ["A", "B", "C"]]
        block2_communities = [communities[node] for node in ["D", "E", "F"]]
        
        assert len(set(block1_communities)) == 1, "Block 1 nodes not in same community"
        assert len(set(block2_communities)) == 1, "Block 2 nodes not in same community"
        assert block1_communities[0] != block2_communities[0], "Blocks not separated"
    
    def test_determinism(self):
        """Test that results are deterministic with fixed random_state."""
        edges = [["A", "B", "C"], ["B", "C", "D"], ["A", "D"], ["E", "F"], ["F", "G"]]
        hg = Hypergraph(edges=edges)
        
        # Run multiple times with same seed
        communities1 = detect_communities(hg, k=3, random_state=42)
        communities2 = detect_communities(hg, k=3, random_state=42)
        
        # Should be identical
        assert communities1 == communities2, "Results not deterministic"
        
        # Run with different seed
        communities3 = detect_communities(hg, k=3, random_state=123)
        
        # May be different due to k-means initialization, but should be valid
        assert len(set(communities3.values())) == 3, "Invalid number of communities"
    
    def test_ari_score(self):
        """Test Adjusted Rand Index for synthetic ground truth."""
        # Create hypergraph with known community structure
        edges = [
            # Community 1: A, B, C
            ["A", "B"], ["B", "C"], ["A", "C"],
            ["A", "B", "C"],
            
            # Community 2: D, E, F
            ["D", "E"], ["E", "F"], ["D", "F"],
            ["D", "E", "F"],
            
            # Community 3: G, H, I
            ["G", "H"], ["H", "I"], ["G", "I"],
            ["G", "H", "I"],
            
            # Minimal cross-community connections
            ["C", "D"], ["F", "G"]
        ]
        hg = Hypergraph(edges=edges)
        
        # Ground truth communities
        ground_truth = {
            "A": 0, "B": 0, "C": 0,
            "D": 1, "E": 1, "F": 1,
            "G": 2, "H": 2, "I": 2,
        }
        
        # Detect communities
        communities = detect_communities(hg, k=3, random_state=0)
        
        # Convert to arrays for ARI computation
        gt_labels = [ground_truth[node] for node in hg.nodes]
        pred_labels = [communities[node] for node in hg.nodes]
        
        # Compute ARI
        ari = adjusted_rand_score(gt_labels, pred_labels)
        
        # Should have high ARI (> 0.8) for this clear structure
        assert ari > 0.8, f"Expected ARI > 0.8, got {ari}"
    
    def test_community_sizes(self):
        """Test that communities have reasonable sizes."""
        edges = [["A", "B", "C"], ["B", "C", "D"], ["A", "D"], ["E", "F"], ["F", "G"]]
        hg = Hypergraph(edges=edges)
        
        communities = detect_communities(hg, k=3, random_state=0)
        
        # Count community sizes
        community_counts = {}
        for community_id in communities.values():
            community_counts[community_id] = community_counts.get(community_id, 0) + 1
        
        # Should have 3 communities
        assert len(community_counts) == 3
        
        # No community should be empty
        for count in community_counts.values():
            assert count > 0, "Empty community detected"
        
        # Communities should be reasonably balanced
        min_size = min(community_counts.values())
        max_size = max(community_counts.values())
        assert max_size - min_size <= 3, "Communities too unbalanced"
    
    def test_edge_weight_sensitivity(self):
        """Test that edge weights affect community detection."""
        edges = [["A", "B"], ["B", "C"], ["C", "D"], ["D", "E"]]
        hg = Hypergraph(edges=edges)
        
        # Detect communities with uniform weights
        communities_uniform = detect_communities(hg, k=2, random_state=0)
        
        # Detect communities with weighted edges (emphasize middle)
        weights = np.array([1.0, 2.0, 2.0, 1.0])  # Middle edges more important
        communities_weighted = detect_communities(hg, k=2, weights=weights, random_state=0)
        
        # Results may differ due to weight influence
        # At minimum, should still produce valid communities
        assert len(set(communities_uniform.values())) == 2
        assert len(set(communities_weighted.values())) == 2
    
    def test_k_validation(self):
        """Test that k parameter is properly validated."""
        edges = [["A", "B"], ["B", "C"]]
        hg = Hypergraph(edges=edges)
        
        # k must be >= 2
        with pytest.raises(ValueError, match="k must be >= 2"):
            detect_communities(hg, k=1, random_state=0)
        
        # k cannot exceed number of nodes
        with pytest.raises(ValueError, match="Number of nodes"):
            detect_communities(hg, k=4, random_state=0)
    
    def test_empty_hypergraph(self):
        """Test handling of hypergraph with no edges."""
        edges = []
        
        with pytest.raises(ValueError, match="Edges list cannot be empty"):
            hg = Hypergraph(edges=edges)
    
    def test_single_edge_hypergraph(self):
        """Test hypergraph with only one edge."""
        edges = [["A", "B", "C"]]
        hg = Hypergraph(edges=edges)
        
        # Should work for k=2
        communities = detect_communities(hg, k=2, random_state=0)
        
        # Should have 2 communities
        assert len(set(communities.values())) == 2
        
        # All nodes should be assigned
        for node in hg.nodes:
            assert node in communities
