# Choosing the Number of Communities (k)

## The Challenge

Selecting the optimal number of communities `k` is one of the most important decisions in spectral clustering. Unlike supervised learning, we don't have ground truth labels to guide our choice. This guide presents principled approaches based on spectral graph theory.

## Eigengap Heuristic

The **eigengap heuristic** is the most widely used method for choosing `k` in spectral clustering.

### What is the Eigengap?

The eigengap is the difference between consecutive eigenvalues of the normalized Laplacian:

**Eigengap(i) = λ_{i+1} - λ_i**

where λ₁ ≤ λ₂ ≤ ... ≤ λ_n are the eigenvalues.

### The Rule

**Choose k such that λ_k is the last eigenvalue before a significant gap.**

In mathematical terms:
- Look for the largest index `i` where `λ_{i+1} - λ_i` is "large"
- Set `k = i`

### Why This Works

1. **Spectral gap theory**: Large eigengaps indicate natural cluster boundaries
2. **Stability**: Eigenvalues near gaps are more stable to perturbations
3. **Interpretability**: Clear eigengaps suggest well-separated communities

### Example

Consider eigenvalues: [0.0, 0.1, 0.2, 0.8, 1.2, 1.5]

- Eigengap(1) = 0.1 - 0.0 = 0.1
- Eigengap(2) = 0.2 - 0.1 = 0.1  
- Eigengap(3) = 0.8 - 0.2 = **0.6** ← Large gap!
- Eigengap(4) = 1.2 - 0.8 = 0.4
- Eigengap(5) = 1.5 - 1.2 = 0.3

**Recommendation**: Choose `k = 3` because there's a large gap after λ₃.

## Stability-Based Methods

### Multiple Initializations

Run spectral clustering multiple times with different random seeds and compare results:

```python
from hypergraph_spectral_decomposer import detect_communities

# Test different k values
k_candidates = [2, 3, 4, 5]
stability_scores = []

for k in k_candidates:
    # Run multiple times
    results = []
    for seed in range(10):
        communities = detect_communities(hg, k=k, random_state=seed)
        results.append(communities)
    
    # Measure stability (e.g., Adjusted Rand Index between runs)
    stability = compute_stability(results)
    stability_scores.append(stability)

# Choose k with highest stability
optimal_k = k_candidates[np.argmax(stability_scores)]
```

### Cross-Validation Approach

1. **Split edges**: Randomly partition hyperedges into training/validation sets
2. **Train**: Run spectral clustering on training set
3. **Validate**: Evaluate clustering quality on validation set
4. **Repeat**: Test different k values

## Practical Guidelines

### Start with Domain Knowledge

- **Expected structure**: How many communities do you expect?
- **Application context**: What makes sense for your use case?
- **Computational limits**: Larger k means more computation

### Use Multiple Methods

Don't rely on a single method:

1. **Eigengap analysis**: Look for natural breaks
2. **Stability testing**: Check consistency across runs
3. **Domain expertise**: Apply subject matter knowledge
4. **Visualization**: Plot eigenvalues and look for gaps

### Common Patterns

- **k = 2**: Often good for binary classification problems
- **k = 3-5**: Common for social networks, academic fields
- **k = 5-10**: For complex systems with many distinct groups
- **k > 10**: Usually requires very large, well-structured networks

## Implementation in the Library

The library provides tools to help with k selection:

```python
from hypergraph_spectral_decomposer.spectral import (
    compute_normalized_laplacian,
    spectral_embedding
)

# Compute Laplacian
L = compute_normalized_laplacian(hg)

# Get eigenvalues for analysis
eigenvals, _ = spectral_embedding(L, k=min(10, len(hg.nodes)-1))

# Analyze eigengaps
eigengaps = np.diff(eigenvals)
print("Eigengaps:", eigengaps)

# Find largest gap
max_gap_idx = np.argmax(eigengaps)
suggested_k = max_gap_idx + 1
print(f"Suggested k: {suggested_k}")
```

## When to Be Cautious

### No Clear Eigengap

If eigenvalues decrease smoothly without clear gaps:
- The network may not have well-defined community structure
- Consider using a smaller k or alternative methods
- The data might be too noisy or homogeneous

### Multiple Large Gaps

If you see several large eigengaps:
- There might be hierarchical community structure
- Consider multiple levels of clustering
- Use domain knowledge to choose the most relevant level

### Small Network Size

For very small networks (n < 20):
- Eigengap analysis may be unreliable
- Use domain knowledge and visualization
- Consider if clustering is even necessary

## References

For theoretical foundations and advanced methods:

- [von Luxburg (2007)](https://arxiv.org/abs/0711.0189): Comprehensive tutorial on spectral clustering
- [Ng, Jordan, Weiss (2002)](https://ai.stanford.edu/~ang/papers/nips01-spectral.pdf): Theoretical analysis of spectral clustering
- [Chung (1997)](https://bookstore.ams.org/cbms-92/): Classical spectral graph theory

## Summary

Choosing `k` is both an art and a science:

1. **Start with eigengap analysis** - look for natural breaks
2. **Test stability** - run multiple times with different seeds
3. **Apply domain knowledge** - what makes sense for your problem?
4. **Use multiple methods** - don't rely on a single approach
5. **Validate results** - check that communities are meaningful

The optimal `k` balances mathematical rigor with practical interpretability. Trust your data, but also trust your intuition about the underlying structure.
