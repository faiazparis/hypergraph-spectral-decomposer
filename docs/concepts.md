# Hypergraph Spectral Analysis: Core Concepts

## What is a Hypergraph?

A hypergraph is a generalization of a graph where edges (called hyperedges) can connect any number of nodes, not just two. This makes hypergraphs ideal for modeling multi-way relationships like:

- **Teams**: Multiple people working together
- **Group chats**: Multiple participants in conversations  
- **Product bundles**: Multiple items sold together
- **Academic collaborations**: Multiple authors on papers

## Mathematical Representation

### Incidence Matrix H

The core representation is the **incidence matrix** H of size n × m:

- **n**: number of nodes
- **m**: number of hyperedges
- **H[i,j] = 1**: if node i belongs to hyperedge j
- **H[i,j] = 0**: otherwise

### Degree Matrices

Two key diagonal matrices capture the structure:

**Node degree matrix Dv = diag(H 1_m)**
- Dv[i,i] = number of hyperedges containing node i
- Represents how "active" each node is

**Edge size matrix De = diag(1_n^T H)**  
- De[j,j] = number of nodes in hyperedge j
- Represents the "size" of each hyperedge

## Zhou's Normalized Laplacian

The normalized Laplacian L is computed as:

**L = I - S**

where the similarity matrix S is:

**S = Dv^(-1/2) H W De^(-1) H^T Dv^(-1/2)**

### Key Properties

1. **Symmetry**: L = L^T (ensured numerically)
2. **Positive Semi-Definite**: All eigenvalues ≥ 0
3. **Zero eigenvalues**: Multiplicity equals number of connected components
4. **Spectral gap**: Important for community detection quality

### Edge Weights W

The weight matrix W allows customizing hyperedge importance:
- **W = I**: Uniform weights (default)
- **W = diag(w)**: Custom weights per hyperedge
- **W[i,i]**: Importance of hyperedge i

## Spectral Embedding

The spectral embedding finds the k smallest non-trivial eigenvectors of L:

**X = [v₁, v₂, ..., v_k]**

where v_i are eigenvectors corresponding to eigenvalues λ₁ ≤ λ₂ ≤ ... ≤ λ_k.

### Why This Works

1. **Low-dimensional representation**: Projects nodes to k-dimensional space
2. **Structure preservation**: Similar nodes stay close in embedding
3. **Community structure**: Natural clusters emerge in spectral space

## Community Detection Pipeline

1. **Build hypergraph** from CSV or edge lists
2. **Compute Laplacian** L using Zhou's formula
3. **Find eigenvectors** of L (spectral embedding)
4. **Apply k-means** to cluster nodes in spectral space
5. **Map back** to original node IDs

## Numerical Stability

The implementation includes several stability measures:

- **Epsilon threshold**: Prevents division by zero in inverses
- **Symmetry enforcement**: (S + S^T)/2 for numerical precision
- **Disconnected handling**: Robust detection of λ=0 eigenvectors
- **Sparse operations**: Efficient computation for large hypergraphs

## References

For detailed mathematical proofs and theoretical foundations, see:

- [Zhou et al. (2006)](https://papers.nips.cc/paper_files/paper/2006/file/dff8e9c2ac33381546d96deea9922999-Paper.pdf): Original normalized Laplacian formulation
- [von Luxburg (2007)](https://arxiv.org/abs/0711.0189): Spectral clustering theory
- [Chung (1997)](https://bookstore.ams.org/cbms-92/): Classical spectral graph theory
