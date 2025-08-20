"""Test command-line interface functionality."""

import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from hypergraph_spectral_decomposer.cli import app


class TestCLIFunctionality:
    """Test CLI commands and functionality."""
    
    def setup_method(self):
        """Set up CLI runner for each test."""
        self.runner = CliRunner()
    
    def test_detect_command_help(self):
        """Test that detect command shows help."""
        result = self.runner.invoke(app, ["detect", "--help"])
        assert result.exit_code == 0
        assert "detect" in result.stdout
        assert "input_path" in result.stdout
        assert "--k" in result.stdout
        assert "--output" in result.stdout
    
    def test_detect_command_basic(self, tmp_path):
        """Test basic community detection workflow."""
        # Create test CSV
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("A,B\nB,C\nA,C")
        
        output_csv = tmp_path / "output.csv"
        
        # Run command
        result = self.runner.invoke(
            app,
            [
                "detect",
                str(input_csv),
                "--k", "2",
                "--output", str(output_csv),
            ],
        )
        
        assert result.exit_code == 0
        assert "Successfully wrote" in result.stdout
        
        # Check output file exists and has correct schema
        assert output_csv.exists()
        
        # Read and validate output
        df = pd.read_csv(output_csv)
        assert list(df.columns) == ["node", "community"]
        assert len(df) == 3  # 3 nodes
        
        # Check that we got 2 communities
        unique_communities = set(df["community"])
        assert len(unique_communities) == 2
    
    def test_detect_command_with_header(self, tmp_path):
        """Test CSV loading with header row."""
        # Create test CSV with header
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("node1,node2,node3\nA,B\nB,C,D\nA,C")
        
        output_csv = tmp_path / "output.csv"
        
        # Run command with header flag
        result = self.runner.invoke(
            app,
            [
                "detect",
                str(input_csv),
                "--k", "2",
                "--output", str(output_csv),
                "--has-header",
            ],
        )
        
        assert result.exit_code == 0
        assert "Successfully wrote" in result.stdout
        
        # Check output
        df = pd.read_csv(output_csv)
        assert len(df) == 4  # A, B, C, D
        assert len(set(df["community"])) == 2
    
    def test_detect_command_custom_delimiter(self, tmp_path):
        """Test CSV loading with custom delimiter."""
        # Create test CSV with semicolon delimiter
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("A;B\nB;C;D\nA;C")
        
        output_csv = tmp_path / "output.csv"
        
        # Run command with custom delimiter
        result = self.runner.invoke(
            app,
            [
                "detect",
                str(input_csv),
                "--k", "2",
                "--output", str(output_csv),
                "--delimiter", ";",
            ],
        )
        
        assert result.exit_code == 0
        assert "Successfully wrote" in result.stdout
        
        # Check output
        df = pd.read_csv(output_csv)
        assert len(df) == 4  # A, B, C, D
        assert len(set(df["community"])) == 2
    
    def test_detect_command_with_weights(self, tmp_path):
        """Test community detection with edge weights."""
        # Create test CSV
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("A,B\nB,C\nA,C")
        
        # Create weights file
        weights_csv = tmp_path / "weights.csv"
        weights_csv.write_text("2.0\n1.0\n1.5")
        
        output_csv = tmp_path / "output.csv"
        
        # Run command with weights
        result = self.runner.invoke(
            app,
            [
                "detect",
                str(input_csv),
                "--k", "2",
                "--output", str(output_csv),
                "--weights", str(weights_csv),
            ],
        )
        
        assert result.exit_code == 0
        assert "Successfully wrote" in result.stdout
        
        # Check output
        df = pd.read_csv(output_csv)
        assert len(df) == 3
        assert len(set(df["community"])) == 2
    
    def test_detect_command_with_plot(self, tmp_path):
        """Test community detection with plot generation."""
        # Create test CSV
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("A,B\nB,C\nA,C")
        
        output_csv = tmp_path / "output.csv"
        plot_png = tmp_path / "plot.png"
        
        # Run command with plot
        result = self.runner.invoke(
            app,
            [
                "detect",
                str(input_csv),
                "--k", "2",
                "--output", str(output_csv),
                "--plot", str(plot_png),
            ],
        )
        
        assert result.exit_code == 0
        assert "Plot generated successfully" in result.stdout
        
        # Check that plot file was created
        assert plot_png.exists()
        assert plot_png.stat().st_size > 0  # Not empty
    
    def test_detect_command_random_state(self, tmp_path):
        """Test that random_state produces deterministic results."""
        # Create test CSV
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("A,B\nB,C\nA,C\nD,E\nE,F\nD,F")
        
        output_csv1 = tmp_path / "output1.csv"
        output_csv2 = tmp_path / "output2.csv"
        
        # Run command twice with same random_state
        result1 = self.runner.invoke(
            app,
            [
                "detect",
                str(input_csv),
                "--k", "2",
                "--output", str(output_csv1),
                "--random-state", "42",
            ],
        )
        
        result2 = self.runner.invoke(
            app,
            [
                "detect",
                str(input_csv),
                "--k", "2",
                "--output", str(output_csv2),
                "--random-state", "42",
            ],
        )
        
        assert result1.exit_code == 0
        assert result2.exit_code == 0
        
        # Results should be identical
        df1 = pd.read_csv(output_csv1)
        df2 = pd.read_csv(output_csv2)
        
        # Sort by node for comparison
        df1 = df1.sort_values("node").reset_index(drop=True)
        df2 = df2.sort_values("node").reset_index(drop=True)
        
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_detect_command_validation(self, tmp_path):
        """Test input validation in CLI."""
        # Test invalid k
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("A,B\nB,C")
        
        output_csv = tmp_path / "output.csv"
        
        result = self.runner.invoke(
            app,
            [
                "detect",
                str(input_csv),
                "--k", "3",  # k > n_nodes
                "--output", str(output_csv),
            ],
        )
        
        assert result.exit_code == 1
        assert "Error:" in result.stderr
    
    def test_detect_command_nonexistent_file(self):
        """Test handling of nonexistent input file."""
        output_csv = Path("nonexistent_output.csv")
        
        result = self.runner.invoke(
            app,
            [
                "detect",
                "nonexistent.csv",
                "--k", "2",
                "--output", str(output_csv),
            ],
        )
        
        assert result.exit_code == 1
        assert "Error:" in result.stderr
    
    def test_detect_command_invalid_weights(self, tmp_path):
        """Test handling of invalid weights file."""
        # Create test CSV
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("A,B\nB,C\nA,C")
        
        # Create invalid weights file (wrong number of weights)
        weights_csv = tmp_path / "weights.csv"
        weights_csv.write_text("1.0\n2.0")  # Only 2 weights for 3 edges
        
        output_csv = tmp_path / "output.csv"
        
        result = self.runner.invoke(
            app,
            [
                "detect",
                str(input_csv),
                "--k", "2",
                "--output", str(output_csv),
                "--weights", str(weights_csv),
            ],
        )
        
        assert result.exit_code == 1
        assert "Error:" in result.stderr
    
    def test_detect_command_n_init_parameter(self, tmp_path):
        """Test n_init parameter for k-means."""
        # Create test CSV
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("A,B\nB,C\nA,C")
        
        output_csv = tmp_path / "output.csv"
        
        # Run command with custom n_init
        result = self.runner.invoke(
            app,
            [
                "detect",
                str(input_csv),
                "--k", "2",
                "--output", str(output_csv),
                "--n-init", "20",
            ],
        )
        
        assert result.exit_code == 0
        assert "Successfully wrote" in result.stdout
        
        # Check output
        df = pd.read_csv(output_csv)
        assert len(df) == 3
        assert len(set(df["community"])) == 2
