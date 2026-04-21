#!/bin/bash
# Reproduce all experiments in Section 7.4 of the paper.

set -e

echo "=== Step 1: Observation Analysis (Figure 7-9, Figure 7-10) ==="
python run_observation_analysis.py

echo "=== Step 2: Intervention Analysis (Figure 7-11) ==="
python run_intervention_analysis.py

echo "=== Step 3: Mechanism Analysis (Figure 7-12) ==="
python run_mechanism_analysis.py

echo "=== Step 4: Aggregate Core Figure ==="
python generate_core_figure.py

echo "All outputs are saved in analysis_results/"
