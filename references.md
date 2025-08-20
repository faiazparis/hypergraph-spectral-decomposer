# References and Mathematical Foundation

## Key Mathematical Formulas

### Degree Matrices
- **Node degree matrix**: Dv = diag(H 1_m)
- **Edge size matrix**: De = diag(1_n^T H)

### Normalized Laplacian
- **Similarity matrix**: S = Dv^(-1/2) H W De^(-1) H^T Dv^(-1/2)
- **Normalized Laplacian**: L = I - S

where:
- H: incidence matrix (n × m)
- Dv: node degree matrix
- De: edge size matrix  
- W: edge weight matrix (default: identity)
- I: identity matrix

## Citations

### Core Spectral Hypergraph Theory

**Zhou, Huang, Schölkopf (2006). Learning with Hypergraphs.**
- **Conference**: NeurIPS 2006
- **URL**: https://papers.nips.cc/paper_files/paper/2006/file/dff8e9c2ac33381546d96deea9922999-Paper.pdf
- **Key contribution**: Introduced the normalized Laplacian for hypergraphs, establishing the mathematical foundation for spectral hypergraph analysis.

### Spectral Clustering Theory

**von Luxburg (2007). A Tutorial on Spectral Clustering.**
- **arXiv**: 0711.0189
- **URL**: https://arxiv.org/abs/0711.0189
- **Key contribution**: Comprehensive tutorial on spectral clustering theory, including eigengap analysis and stability considerations.

**Ng, Jordan, Weiss (2002). On Spectral Clustering: Analysis and an Algorithm.**
- **Conference**: NeurIPS 2001
- **URL**: https://ai.stanford.edu/~ang/papers/nips01-spectral.pdf
- **Key contribution**: Established theoretical foundations for spectral clustering, including convergence guarantees and algorithm analysis.

### Hypergraph Spectral Methods

**Chan, Louis (2018). Spectral Clustering of Graphs and Hypergraphs.**
- **arXiv**: 1808.07464
- **URL**: https://arxiv.org/abs/1808.07464
- **Key contribution**: Extended spectral clustering to hypergraphs with theoretical analysis and practical algorithms.

**Tudisco, Higham (2019). A Nonlinear Spectral Method for Hypergraph Clustering.**
- **arXiv**: 1806.04792
- **URL**: https://arxiv.org/abs/1806.04792
- **Key contribution**: Introduced nonlinear spectral methods for hypergraph clustering, providing alternative approaches to the linear Laplacian.

### Hypergraph Theory

**Banerjee, Char, Chandr (2017). Spectra of Hypergraphs.**
- **arXiv**: 1701.08134
- **URL**: https://arxiv.org/abs/1701.08134
- **Key contribution**: Comprehensive study of hypergraph spectra and their properties.

**Chitra, Raphael (2019). Random Walks on Hypergraphs with Edge-Dependent Vertex Weights.**
- **arXiv**: 1905.03242
- **URL**: https://arxiv.org/abs/1905.03242
- **Key contribution**: Advanced random walk models for hypergraphs with sophisticated weighting schemes.

### Classical Spectral Graph Theory

**Chung (1997). Spectral Graph Theory.**
- **Publisher**: American Mathematical Society
- **Series**: CBMS Regional Conference Series in Mathematics, Number 92
- **URL**: https://bookstore.ams.org/cbms-92/
- **Key contribution**: Foundational text on spectral graph theory, providing the theoretical basis for all spectral methods in network analysis.

## Implementation Notes

The library implements Zhou's normalized Laplacian approach as the primary method, with numerical stability considerations:

- Small epsilon (ε) for inverse operations to prevent division by zero
- Symmetry enforcement for the similarity matrix S
- Robust handling of disconnected components in spectral embedding
- Sparse matrix operations for computational efficiency

All mathematical operations are validated through deterministic tests that verify algebraic identities and theoretical properties.
