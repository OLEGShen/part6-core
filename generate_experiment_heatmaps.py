"""Semantic entrypoint for experiment heatmap generation."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    legacy_script = Path(__file__).with_name("4_generate_experiment_heatmaps.py")
    runpy.run_path(str(legacy_script), run_name="__main__")
