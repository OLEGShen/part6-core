"""Semantic entrypoint for LLM-based thought keyword extraction."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    legacy_script = Path(__file__).with_name("1_analyze_thoughts_with_llm.py")
    runpy.run_path(str(legacy_script), run_name="__main__")
