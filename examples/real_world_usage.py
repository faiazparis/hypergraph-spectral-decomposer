#!/usr/bin/env python3
"""
Real-World Hypergraph Community Detection Example

This example shows how to use the library with your own data and encourages
community development of new applications.

Mission: Build Advanced Math Models for Everyone
"""

import numpy as np
import pandas as pd
from pathlib import Path
from hypergraph_spectral_decomposer import Hypergraph, detect_communities


def analyze_social_network():
    """
    Example: Social Network Analysis
    
    Analyze a social network where people participate in various groups
    (teams, clubs, events, etc.) to discover natural communities.
    """
    print("🔍 Social Network Community Detection")
    print("=" * 50)
    
    # Example: People participating in different groups
    social_edges = [
        ["Alice", "Bob", "Charlie"],      # Team Alpha
        ["Bob", "Charlie", "David"],      # Team Beta  
        ["Alice", "Eve", "Frank"],        # Club Gamma
        ["Eve", "Frank", "Grace"],        # Club Delta
        ["David", "Grace", "Henry"],      # Event Epsilon
        ["Alice", "David", "Grace"],      # Committee Zeta
    ]
    
    # Create hypergraph
    hg = Hypergraph(edges=social_edges)
    print(f"📊 Hypergraph: {len(hg)} people, {len(hg.edges)} groups")
    
    # Detect communities
    communities = detect_communities(hg, k=3, random_state=42)
    
    # Analyze results
    print("\n🏘️  Detected Communities:")
    for community_id in sorted(set(communities.values())):
        members = [node for node, cid in communities.items() if cid == community_id]
        print(f"  Community {community_id}: {', '.join(members)}")
    
    return hg, communities


def analyze_recommendation_system():
    """
    Example: Recommendation System Analysis
    
    Analyze user-item interactions to discover user communities
    based on similar preferences.
    """
    print("\n🛒 Recommendation System Analysis")
    print("=" * 50)
    
    # Example: Users and items they've interacted with
    user_item_edges = [
        ["User1", "MovieA", "MovieB", "MovieC"],    # Action movie fan
        ["User2", "MovieA", "MovieB", "MovieD"],    # Action movie fan
        ["User3", "MovieE", "MovieF", "MovieG"],    # Romance fan
        ["User4", "MovieE", "MovieF", "MovieH"],    # Romance fan
        ["User5", "MovieA", "MovieE", "MovieI"],    # Mixed preferences
        ["User6", "MovieB", "MovieF", "MovieJ"],    # Mixed preferences
    ]
    
    # Create hypergraph
    hg = Hypergraph(edges=user_item_edges)
    print(f"📊 Hypergraph: {len(hg)} users, {len(hg.edges)} item groups")
    
    # Detect communities
    communities = detect_communities(hg, k=2, random_state=42)
    
    # Analyze results
    print("\n🏘️  User Communities:")
    for community_id in sorted(set(communities.values())):
        users = [node for node, cid in communities.items() if cid == community_id]
        print(f"  Community {community_id}: {', '.join(users)}")
    
    return hg, communities


def analyze_biological_network():
    """
    Example: Biological Network Analysis
    
    Analyze protein-protein interactions to discover functional modules
    in biological systems.
    """
    print("\n🧬 Biological Network Analysis")
    print("=" * 50)
    
    # Example: Proteins participating in different biological processes
    protein_edges = [
        ["Protein1", "Protein2", "Protein3"],        # Cell cycle
        ["Protein2", "Protein3", "Protein4"],        # Cell cycle
        ["Protein5", "Protein6", "Protein7"],        # Metabolism
        ["Protein6", "Protein7", "Protein8"],        # Metabolism
        ["Protein1", "Protein5", "Protein9"],        # Signaling
        ["Protein3", "Protein7", "Protein10"],       # Cross-pathway
    ]
    
    # Create hypergraph
    hg = Hypergraph(edges=protein_edges)
    print(f"📊 Hypergraph: {len(hg)} proteins, {len(hg.edges)} processes")
    
    # Detect communities
    communities = detect_communities(hg, k=3, random_state=42)
    
    # Analyze results
    print("\n🏘️  Functional Modules:")
    for community_id in sorted(set(communities.values())):
        proteins = [node for node, cid in communities.items() if cid == community_id]
        print(f"  Module {community_id}: {', '.join(proteins)}")
    
    return hg, communities


def save_results_to_csv(hypergraph, communities, filename):
    """Save community detection results to CSV."""
    results = []
    for node in hypergraph.nodes:
        results.append({
            "node": node,
            "community": communities[node],
            "degree": hypergraph.get_node_degree(node)
        })
    
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"💾 Results saved to {filename}")


def main():
    """Run all examples and demonstrate community development potential."""
    print("🚀 Hypergraph Spectral Decomposer - Real-World Examples")
    print("Mission: Build Advanced Math Models for Everyone")
    print("=" * 70)
    
    # Run examples
    social_hg, social_communities = analyze_social_network()
    save_results_to_csv(social_hg, social_communities, "social_communities.csv")
    
    rec_hg, rec_communities = analyze_recommendation_system()
    save_results_to_csv(rec_hg, rec_communities, "recommendation_communities.csv")
    
    bio_hg, bio_communities = analyze_biological_network()
    save_results_to_csv(bio_hg, bio_communities, "biological_communities.csv")
    
    print("\n" + "=" * 70)
    print("🎯 Current Status & Improvement Opportunities:")
    print("  ✅ Core math working correctly")
    print("  ✅ Small to medium hypergraphs (up to ~1000 nodes)")
    print("  ✅ Deterministic results")
    print("\n🔧 What needs work for production:")
    print("  • Scalability: Handle 10K+ node hypergraphs")
    print("  • Robustness: Messy data, edge cases")
    print("  • Performance: Parallel processing, GPU acceleration")
    print("  • Monitoring: Logging, error handling")
    print("\n💡 Bring your own data and help make it production-ready!")
    print("   Current library: Good for research, needs work for scale")


if __name__ == "__main__":
    main()
