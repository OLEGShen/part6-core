#!/bin/bash
# Reproduce all experiments in Section 7.4 of the paper.

set -e

if [ -z "${DASHSCOPE_API_KEY}" ]; then
  echo "Error: DASHSCOPE_API_KEY is not set."
  echo "Paper reproduction requires LLM mode."
  echo "Please run: export DASHSCOPE_API_KEY=your_key"
  exit 1
fi

echo "=== Step 1: Observation Analysis (Figure 7-9, Figure 7-10) ==="
python run_observation_analysis.py --decision_mode llm

echo "=== Step 2: Intervention Analysis (Figure 7-11) ==="
python run_intervention_analysis.py --decision_mode llm

echo "=== Step 3: Mechanism Analysis (Figure 7-12) ==="
python run_mechanism_analysis.py --decision_mode llm

echo "=== Step 4: Aggregate Core Figure ==="
python generate_core_figure.py --decision_mode llm

echo "All outputs are saved in analysis_results/"
