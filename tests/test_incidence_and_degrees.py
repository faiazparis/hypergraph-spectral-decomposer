"""Test incidence matrix construction and degree computations."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from hypergraph_spectral_decomposer.hypergraph import Hypergraph


class TestIncidenceMatrixConstruction:
    """Test building incidence matrices from edge lists."""
    
    def test_simple_hypergraph(self):
        """Test basic hypergraph with 3 nodes and 2 edges."""
        edges = [["A", "B"], ["B", "C"]]
        hg = Hypergraph(edges=edges)
        
        # Check dimensions
        assert len(hg) == 3
        assert len(hg.edges) == 2
        
        # Check node ordering (stable)
        assert hg.nodes == ["A", "B", "C"]
        
        # Check incidence matrix
        H = hg.incidence_matrix
        assert H.shape == (3, 2)
        
        # Expected incidence matrix:
        #   A B C
        # 1 1 1 0  (edge 0: A-B)
        # 2 0 1 1  (edge 1: B-C)
        expected = np.array([[1, 0], [1, 1], [0, 1]])
        np.testing.assert_array_equal(H.toarray(), expected)
    
    def test_degree_computations(self):
        """Test Dv = H 1_m and De = 1_n^T H formulas."""
        edges = [["A", "B", "C"], ["B", "C"], ["A", "D"]]
        hg = Hypergraph(edges=edges)
        
        H = hg.incidence_matrix
        
        # Test Dv = diag(H 1_m) - node degrees
        node_degrees_expected = H.sum(axis=1).A1
        node_degrees_actual = np.array([hg.get_node_degree(node) for node in hg.nodes])
        np.testing.assert_array_equal(node_degrees_actual, node_degrees_expected)
        
        # Test De = diag(1_n^T H) - edge sizes
        edge_sizes_expected = H.sum(axis=0).A1
        edge_sizes_actual = np.array([hg.get_edge_size(i) for i in range(len(hg.edges))])
        np.testing.assert_array_equal(edge_sizes_actual, edge_sizes_expected)
    
    def test_edge_sizes_method(self):
        """Test edge_sizes() method returns correct mapping."""
        edges = [["A", "B", "C"], ["B", "C"], ["A", "D"]]
        hg = Hypergraph(edges=edges)
        
        edge_sizes = hg.edge_sizes()
        expected = {0: 3, 1: 2, 2: 2}  # edge 0 has 3 nodes, edges 1,2 have 2
        
        assert edge_sizes == expected
    
    def test_node_degrees_method(self):
        """Test node_degrees() method returns correct mapping."""
        edges = [["A", "B", "C"], ["B", "C"], ["A", "D"]]
        hg = Hypergraph(edges=edges)
        
        node_degrees = hg.node_degrees()
        expected = {"A": 2, "B": 2, "C": 2, "D": 1}
        
        assert node_degrees == expected
    
    def test_sparsity_pattern(self):
        """Test that incidence matrix has correct sparsity pattern."""
        edges = [["A", "B"], ["B", "C", "D"], ["A", "C"]]
        hg = Hypergraph(edges=edges)
        
        H = hg.incidence_matrix
        
        # Check sparsity
        assert H.nnz == 7  # 2 + 3 + 2 = 7 total node-edge connections
        
        # Check specific connections
        A_idx = hg.nodes.index("A")
        B_idx = hg.nodes.index("B")
        C_idx = hg.nodes.index("C")
        D_idx = hg.nodes.index("D")
        
        # A in edge 0 and 2
        assert H[A_idx, 0] == 1
        assert H[A_idx, 2] == 1
        assert H[A_idx, 1] == 0
        
        # B in edge 0 and 1
        assert H[B_idx, 0] == 1
        assert H[B_idx, 1] == 1
        assert H[B_idx, 2] == 0


class TestInputValidation:
    """Test input validation and edge cases."""
    
    def test_minimum_nodes_per_edge(self):
        """Test enforcement of minimum nodes per edge."""
        edges = [["A"], ["A", "B"], ["A", "B", "C"]]  # edge 0 has only 1 node
        
        # Should drop edge with < 2 nodes
        hg = Hypergraph(edges=edges, min_nodes_per_edge=2)
        assert len(hg.edges) == 2
        
        # Check that edges contain expected nodes (order may vary due to deduplication)
        edge_nodes = [set(edge) for edge in hg.edges]
        assert {"A", "B"} in edge_nodes
        assert {"A", "B", "C"} in edge_nodes
    
    def test_whitespace_trimming(self):
        """Test that whitespace is trimmed from node IDs."""
        edges = [[" A ", " B "], ["B", " C "]]
        hg = Hypergraph(edges=edges)
        
        assert "A" in hg.nodes
        assert "B" in hg.nodes
        assert "C" in hg.nodes
        assert len(hg.nodes) == 3  # No duplicates
    
    def test_deduplication(self):
        """Test that duplicate nodes within edges are removed."""
        edges = [["A", "B", "A"], ["B", "C", "B"]]  # duplicates within edges
        hg = Hypergraph(edges=edges)
        
        # Should dedupe within each edge
        assert hg.get_edge_size(0) == 2  # A, B (A removed)
        assert hg.get_edge_size(1) == 2  # B, C (B removed)
    
    def test_empty_edges_handling(self):
        """Test handling of empty edge lists."""
        edges = [["A", "B"], [], ["B", "C"]]  # middle edge is empty
        hg = Hypergraph(edges=edges)
        
        # Should skip empty edges
        assert len(hg.edges) == 2
        
        # Check that edges contain expected nodes (order may vary due to deduplication)
        edge_nodes = [set(edge) for edge in hg.edges]
        assert {"A", "B"} in edge_nodes
        assert {"B", "C"} in edge_nodes
    
    def test_stable_node_ordering(self):
        """Test that node ordering is stable across runs."""
        edges = [["C", "A", "B"], ["D", "B", "A"]]
        
        # Multiple constructions should give same ordering
        hg1 = Hypergraph(edges=edges)
        hg2 = Hypergraph(edges=edges)
        
        assert hg1.nodes == hg2.nodes
        # Should be sorted alphabetically
        assert hg1.nodes == ["A", "B", "C", "D"]


class TestCSVLoading:
    """Test loading hypergraphs from CSV files."""
    
    def test_csv_without_header(self, tmp_path):
        """Test CSV loading without header row."""
        # Create CSV with consistent column count by padding with empty strings
        csv_content = "A,B,\nB,C,D\nA,C,"
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)
        
        hg = Hypergraph(csv_path=str(csv_file), has_header=False)
        
        assert len(hg) == 4  # A, B, C, D
        assert len(hg.edges) == 3
        # Check that edges contain expected nodes (order may vary due to deduplication)
        edge_nodes = [set(edge) for edge in hg.edges]
        assert {"A", "B"} in edge_nodes
        assert {"B", "C", "D"} in edge_nodes
        assert {"A", "C"} in edge_nodes
    
    def test_csv_with_header(self, tmp_path):
        """Test CSV loading with header row."""
        csv_content = "node1,node2,node3\nA,B\nB,C,D\nA,C"
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)
        
        hg = Hypergraph(csv_path=str(csv_file), has_header=True)
        
        assert len(hg) == 4  # A, B, C, D
        assert len(hg.edges) == 3
        # Header row should be ignored
        assert ["A", "B"] in hg.edges
        assert ["B", "C", "D"] in hg.edges
        assert ["A", "C"] in hg.edges
    
    def test_csv_delimiter(self, tmp_path):
        """Test CSV loading with custom delimiter."""
        # Create CSV with consistent column count by padding with empty strings
        csv_content = "A;B;\nB;C;D\nA;C;"
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)
        
        hg = Hypergraph(csv_path=str(csv_file), delimiter=";")
        
        assert len(hg) == 4
        assert len(hg.edges) == 3
