# Hypergraph Spectral Decomposer

**Mission: Build advanced math models for everyone.**

## What it solves

This library detects communities in hypergraphs—networks where relationships involve multiple nodes simultaneously (teams, group chats, product bundles). Unlike traditional graphs limited to pairwise connections, hypergraphs capture the rich, multi-way interactions that dominate real-world systems.

**Key capabilities:**
- **Robust community detection** in higher-order, sparse, possibly disconnected hypergraphs
- **Reproducible pipeline** from CSV hyperedges to community assignments
- **Spectral approach** using Zhou's normalized Laplacian for mathematical rigor
- **Local-only operation** with no external dependencies or paid services

## Quickstart

### Install locally

```bash
# Using Poetry (recommended)
poetry install

# Or using pip
pip install -e ".[dev]"
```

### Run community detection

```bash
hgsd detect --input path/to/hyperedges.csv --k 3 --output communities.csv
```

### Run tests

```bash
# Install pre-commit hooks
pre-commit install

# Run test suite with coverage
pytest --cov=src --cov-report=term-missing
```

## Input format

Your CSV should have one hyperedge per row, with node IDs in each cell:

```csv
node1,node2,node3
user1,user2,user3,user4
item1,item2
```

## Mathematical foundation

The library implements Zhou's normalized Laplacian approach:
- **Incidence matrix H**: Maps nodes to hyperedges
- **Normalized Laplacian L**: L = I - Dv^(-1/2) H W De^(-1) H^T Dv^(-1/2)
- **Spectral embedding**: Eigenvectors of L provide low-dimensional representations
- **K-means clustering**: Groups nodes based on spectral coordinates

## Current Status & Limitations

**What works well:**
- Core mathematical implementation (Zhou's normalized Laplacian)
- Small to medium hypergraphs (up to ~1000 nodes)
- Deterministic results with proper validation
- 74% test coverage on core functionality

**What needs improvement for production:**
- **Scalability**: Memory usage grows with hypergraph size
- **Error handling**: Limited recovery from malformed input
- **Performance**: No parallel processing or GPU acceleration
- **Robustness**: Assumes clean, well-formed data
- **Monitoring**: No logging or performance metrics

**For production use, you'll likely need to:**
- Add proper error handling and validation
- Implement memory-efficient sparse operations
- Add parallel processing for large hypergraphs
- Include monitoring and logging
- Handle edge cases (disconnected components, degenerate cases)

## Contributing

**Building advanced math models for everyone - one step at a time.**

This is an open source project implementing Zhou's normalized Laplacian for spectral hypergraph community detection. We're making the core theory accessible, but there's plenty of room for improvement.

### 🎯 **What You Can Contribute**
- **Production readiness**: Error handling, logging, monitoring
- **Scalability**: Memory optimization, parallel processing, GPU support
- **Real-world robustness**: Handle messy data, missing values, edge cases
- **Performance**: Benchmark against other methods, optimize bottlenecks
- **Documentation**: Clarify complex concepts, add tutorials

### 🧮 **Mathematical Rigor**
Every contribution maintains our commitment to mathematical rigor:
- All methods are theoretically grounded and properly cited
- Comprehensive testing validates algebraic identities
- Clear documentation explains assumptions and limitations

### 📚 **Getting Started**
1. **Understand the math**: Read `docs/concepts.md` and `references.md`
2. **Explore examples**: Run `examples/real_world_usage.py` with your data
3. **Contribute**: Fix bugs, add features, improve documentation
4. **Share**: Use the library in your research, papers, and applications

**Current status**: We have the core math working correctly with 74% test coverage. The library handles small to medium hypergraphs well, but needs work for production use at scale.

See `docs/contributing.md` for detailed guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.
