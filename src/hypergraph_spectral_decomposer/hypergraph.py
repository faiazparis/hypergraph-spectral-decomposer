"""Hypergraph representation with incidence matrix and degree computations."""

from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class Hypergraph:
    """
    Hypergraph representation with incidence matrix and degree computations.
    
    Implements the mathematical foundation from Zhou et al. (2006) for spectral
    hypergraph analysis.
    
    References:
        Zhou, Huang, Schölkopf (2006). Learning with Hypergraphs.
        https://papers.nips.cc/paper_files/paper/2006/file/dff8e9c2ac33381546d96deea9922999-Paper.pdf
    """
    
    def __init__(
        self,
        edges: Optional[List[List[str]]] = None,
        csv_path: Optional[str] = None,
        delimiter: str = ",",
        has_header: bool = False,
        min_nodes_per_edge: int = 2,
    ) -> None:
        """
        Initialize hypergraph from edges or CSV file.
        
        Args:
            edges: List of hyperedges, each a list of node IDs
            csv_path: Path to CSV file with hyperedges (one per row)
            delimiter: CSV delimiter character
            has_header: Whether CSV has header row
            min_nodes_per_edge: Minimum nodes required per hyperedge
            
        Raises:
            ValueError: If invalid input or insufficient nodes per edge
        """
        if edges is not None and csv_path is not None:
            raise ValueError("Cannot specify both edges and csv_path")
        
        if edges is not None:
            self._build_from_edges(edges, min_nodes_per_edge)
        elif csv_path is not None:
            self._build_from_csv(csv_path, delimiter, has_header, min_nodes_per_edge)
        else:
            raise ValueError("Must specify either edges or csv_path")
    
    def _build_from_edges(
        self, edges: List[List[str]], min_nodes_per_edge: int
    ) -> None:
        """Build hypergraph from list of edges."""
        if not edges:
            raise ValueError("Edges list cannot be empty")
        
        # Clean and validate edges
        cleaned_edges = []
        for edge in edges:
            if not edge:
                continue
            
            # Clean node IDs and remove duplicates
            clean_edge = list(set(node.strip() for node in edge if node.strip()))
            
            if len(clean_edge) >= min_nodes_per_edge:
                cleaned_edges.append(clean_edge)
        
        if not cleaned_edges:
            raise ValueError(f"No valid edges with >= {min_nodes_per_edge} nodes")
        
        # Build node mapping and incidence matrix
        self._build_incidence_matrix(cleaned_edges)
    
    def _build_from_csv(
        self,
        csv_path: str,
        delimiter: str,
        has_header: bool,
        min_nodes_per_edge: int,
    ) -> None:
        """Build hypergraph from CSV file."""
        try:
            df = pd.read_csv(csv_path, delimiter=delimiter, header=0 if has_header else None)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")
        
        # Convert to list of edges
        edges = []
        for _, row in df.iterrows():
            edge = [str(cell).strip() for cell in row if pd.notna(cell) and str(cell).strip()]
            if len(edge) >= min_nodes_per_edge:
                edges.append(edge)
        
        if not edges:
            raise ValueError(f"No valid edges with >= {min_nodes_per_edge} nodes")
        
        self._build_incidence_matrix(edges)
    
    def _build_incidence_matrix(self, edges: List[List[str]]) -> None:
        """Build sparse incidence matrix and node/edge mappings."""
        # Create stable node ordering
        all_nodes = set()
        for edge in edges:
            all_nodes.update(edge)
        
        self._nodes = sorted(all_nodes)
        self._node_to_idx = {node: idx for idx, node in enumerate(self._nodes)}
        
        # Build incidence matrix H (n × m)
        n_nodes = len(self._nodes)
        n_edges = len(edges)
        
        # CSR matrix construction
        row_indices = []
        col_indices = []
        
        for edge_idx, edge in enumerate(edges):
            for node in edge:
                row_indices.append(self._node_to_idx[node])
                col_indices.append(edge_idx)
        
        data = np.ones(len(row_indices), dtype=np.float64)
        self._incidence_matrix = csr_matrix(
            (data, (row_indices, col_indices)), shape=(n_nodes, n_edges)
        )
        
        self._edges = edges
        self._edge_sizes = [len(edge) for edge in edges]
    
    @property
    def nodes(self) -> List[str]:
        """Get list of nodes in stable order."""
        return self._nodes.copy()
    
    @property
    def edges(self) -> List[List[str]]:
        """Get list of hyperedges."""
        return [edge.copy() for edge in self._edges]
    
    @property
    def incidence_matrix(self) -> csr_matrix:
        """Get sparse incidence matrix H (n × m)."""
        return self._incidence_matrix.copy()
    
    def node_degrees(self) -> Dict[str, int]:
        """Get node degrees (Dv = diag(H 1_m))."""
        degrees = self._incidence_matrix.sum(axis=1).A1
        return {node: int(deg) for node, deg in zip(self._nodes, degrees)}
    
    def edge_sizes(self) -> Dict[int, int]:
        """Get edge sizes (De = diag(1_n^T H))."""
        sizes = self._incidence_matrix.sum(axis=0).A1
        return {i: int(size) for i, size in enumerate(sizes)}
    
    def get_node_degree(self, node: str) -> int:
        """Get degree of specific node."""
        if node not in self._node_to_idx:
            raise ValueError(f"Node '{node}' not in hypergraph")
        return int(self._incidence_matrix[self._node_to_idx[node], :].sum())
    
    def get_edge_size(self, edge_idx: int) -> int:
        """Get size of specific edge."""
        if edge_idx < 0 or edge_idx >= len(self._edges):
            raise ValueError(f"Edge index {edge_idx} out of range")
        return int(self._incidence_matrix[:, edge_idx].sum())
    
    def __len__(self) -> int:
        """Number of nodes."""
        return len(self._nodes)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Hypergraph({len(self._nodes)} nodes, {len(self._edges)} edges)"
