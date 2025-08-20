"""Command-line interface for hypergraph spectral community detection."""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer
from matplotlib import pyplot as plt

from .hypergraph import Hypergraph
from .community import detect_communities

app = typer.Typer(
    name="hgsd",
    help="Hypergraph Spectral Decomposer - Community detection via Zhou's normalized Laplacian",
)


@app.command()
def detect(
    input_path: Path = typer.Argument(..., help="Path to CSV file with hyperedges"),
    k: int = typer.Option(..., "--k", "-k", help="Number of communities to detect"),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output CSV path"),
    delimiter: str = typer.Option(",", "--delimiter", "-d", help="CSV delimiter"),
    has_header: bool = typer.Option(False, "--has-header", help="CSV has header row"),
    weights_path: Optional[Path] = typer.Option(None, "--weights", "-w", help="Path to edge weights file"),
    plot_path: Optional[Path] = typer.Option(None, "--plot", "-p", help="Path for embedding plot (PNG)"),
    random_state: int = typer.Option(0, "--random-state", help="Random seed for reproducibility"),
    n_init: int = typer.Option(10, "--n-init", help="Number of k-means initializations"),
) -> None:
    """
    Detect communities in hypergraph from CSV file.
    
    The input CSV should have one hyperedge per row, with node IDs in each cell.
    Each hyperedge must have at least 2 nodes.
    """
    try:
        # Load hypergraph
        typer.echo(f"Loading hypergraph from {input_path}")
        hypergraph = Hypergraph(
            csv_path=str(input_path),
            delimiter=delimiter,
            has_header=has_header,
        )
        typer.echo(f"Loaded hypergraph with {len(hypergraph)} nodes and {len(hypergraph.edges)} edges")
        
        # Load weights if provided
        weights = None
        if weights_path is not None:
            typer.echo(f"Loading edge weights from {weights_path}")
            try:
                weights_df = pd.read_csv(weights_path, header=None)
                weights = weights_df.iloc[:, 0].values
                typer.echo(f"Loaded {len(weights)} edge weights")
            except Exception as e:
                typer.echo(f"Error loading weights: {e}", err=True)
                sys.exit(1)
        
        # Validate k
        if k < 2:
            typer.echo("Error: k must be >= 2", err=True)
            sys.exit(1)
        
        if k > len(hypergraph):
            typer.echo(f"Error: k ({k}) cannot exceed number of nodes ({len(hypergraph)})", err=True)
            sys.exit(1)
        
        # Detect communities
        typer.echo(f"Detecting {k} communities...")
        communities = detect_communities(
            hypergraph=hypergraph,
            k=k,
            weights=weights,
            random_state=random_state,
            n_init=n_init,
        )
        
        # Write results
        typer.echo(f"Writing communities to {output_path}")
        results_df = pd.DataFrame([
            {"node": node, "community": community}
            for node, community in communities.items()
        ])
        results_df.to_csv(output_path, index=False)
        typer.echo(f"Successfully wrote {len(communities)} node-community assignments")
        
        # Generate plot if requested
        if plot_path is not None:
            typer.echo(f"Generating embedding plot to {plot_path}")
            _generate_embedding_plot(hypergraph, communities, plot_path)
            typer.echo("Plot generated successfully")
        
        typer.echo("Community detection completed successfully!")
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _generate_embedding_plot(
    hypergraph: Hypergraph, communities: dict, plot_path: Path
) -> None:
    """Generate 2D embedding plot of the communities."""
    from .spectral import compute_normalized_laplacian, spectral_embedding
    
    # Compute Laplacian and embedding
    L = compute_normalized_laplacian(hypergraph)
    k = len(set(communities.values()))
    eigenvals, eigenvecs = spectral_embedding(L, k, random_state=0)
    
    # Use first two non-trivial eigenvectors for 2D plot
    if eigenvecs.shape[1] >= 2:
        x_coords = eigenvecs[:, 0]
        y_coords = eigenvecs[:, 1]
    else:
        # Fallback: use first eigenvector and zeros
        x_coords = eigenvecs[:, 0]
        y_coords = np.zeros_like(x_coords)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot each community with different color
    unique_communities = sorted(set(communities.values()))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_communities)))
    
    for i, community_id in enumerate(unique_communities):
        mask = [communities[node] == community_id for node in hypergraph.nodes]
        plt.scatter(
            x_coords[mask],
            y_coords[mask],
            c=[colors[i]],
            label=f"Community {community_id}",
            alpha=0.7,
            s=50,
        )
    
    plt.xlabel("First eigenvector")
    plt.ylabel("Second eigenvector")
    plt.title(f"Spectral Embedding: {k} Communities")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Main entry point for the CLI."""
    app()

if __name__ == "__main__":
    main()
