#!/usr/bin/env python3
"""
Basic usage example for hypergraph spectral community detection.

This example demonstrates:
1. Creating a hypergraph from edge lists
2. Computing communities using spectral clustering
3. Analyzing the results
"""

import numpy as np
from hypergraph_spectral_decomposer import Hypergraph, detect_communities


def main():
    """Run basic community detection example."""
    print("Hypergraph Spectral Community Detection Example")
    print("=" * 50)
    
    # Create a hypergraph with clear community structure
    edges = [
        # Community 1: A, B, C (tightly connected)
        ["A", "B"], ["B", "C"], ["C", "A"],
        ["A", "B", "C"],  # 3-way connection
        
        # Community 2: D, E, F (tightly connected)
        ["D", "E"], ["E", "F"], ["F", "D"],
        ["D", "E", "F"],  # 3-way connection
        
        # Few cross-community connections
        ["B", "D"], ["C", "E"]
    ]
    
    print(f"Creating hypergraph with {len(edges)} hyperedges...")
    hg = Hypergraph(edges=edges)
    
    print(f"Hypergraph has {len(hg)} nodes: {hg.nodes}")
    print(f"Node degrees: {hg.node_degrees()}")
    print(f"Edge sizes: {hg.edge_sizes()}")
    
    # Detect communities
    print("\nDetecting communities...")
    communities = detect_communities(hg, k=2, random_state=42)
    
    # Display results
    print("\nCommunity assignments:")
    for node in sorted(hg.nodes):
        community = communities[node]
        print(f"  {node} → Community {community}")
    
    # Analyze community structure
    community_nodes = {}
    for node, community in communities.items():
        if community not in community_nodes:
            community_nodes[community] = []
        community_nodes[community].append(node)
    
    print("\nCommunity composition:")
    for community_id, nodes in sorted(community_nodes.items()):
        print(f"  Community {community_id}: {', '.join(sorted(nodes))}")
    
    # Check if communities match expected structure
    expected_community1 = {"A", "B", "C"}
    expected_community2 = {"D", "E", "F"}
    
    actual_community1 = set(community_nodes[0])
    actual_community2 = set(community_nodes[1])
    
    print("\nCommunity quality:")
    print(f"  Community 1 accuracy: {len(actual_community1 & expected_community1)}/3")
    print(f"  Community 2 accuracy: {len(actual_community2 & expected_community2)}/3")
    
    # Test with different k values
    print("\nTesting different k values:")
    for k in [2, 3, 4]:
        if k <= len(hg.nodes):
            communities_k = detect_communities(hg, k=k, random_state=42)
            unique_communities = len(set(communities_k.values()))
            print(f"  k={k}: {unique_communities} communities detected")


if __name__ == "__main__":
    main()
