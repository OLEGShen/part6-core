"""Mechanism analysis pipeline corresponding to Algorithm 7-3."""

import argparse
import json
import os
import urllib.request
import re
from collections import Counter, defaultdict
from pathlib import Path
import hashlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from config import ANALYSIS_RESULTS_DIR, LLM_CONFIG

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional at runtime
    OpenAI = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional at runtime
    SentenceTransformer = None


OUTPUT_DIR = ANALYSIS_RESULTS_DIR
PHASE_LABELS = ["Rule Following", "Anxiety Driven", "Risk Avoidance"]
_FALLBACK_LOGGED = False
_MECHANISM_LLM_CLIENT = None


def discover_thought_files(patterns=None):
    """Collect thought logs from experiments and simulations."""

    base_dir = Path(__file__).resolve().parent
    if patterns is None:
        patterns = ["exp*/**/*_thought.json", "simulation_results/sim_*/**/*_thought.json"]
    paths = []
    for pattern in patterns:
        paths.extend(base_dir.glob(pattern))
    return sorted(paths)


def _get_llm_client():
    global _MECHANISM_LLM_CLIENT
    if _MECHANISM_LLM_CLIENT is not None:
        return _MECHANISM_LLM_CLIENT
    api_key = os.getenv(LLM_CONFIG.api_key_env)
    if not api_key or OpenAI is None:
        return None
    base_url = os.getenv(LLM_CONFIG.base_url_env, LLM_CONFIG.default_base_url)
    _MECHANISM_LLM_CLIENT = OpenAI(api_key=api_key, base_url=base_url)
    return _MECHANISM_LLM_CLIENT


def _log_fallback_once():
    global _FALLBACK_LOGGED
    if not _FALLBACK_LOGGED:
        print("Mechanism analysis fallback: DASHSCOPE_API_KEY not found or LLM unavailable, using keyword matching.")
        _FALLBACK_LOGGED = True


def _parse_json_like(content):
    content = (content or "").strip()
    if content.startswith("```json"):
        content = content.removeprefix("```json").removesuffix("```").strip()
    try:
        return json.loads(content)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            return json.loads(match.group(0))
        raise


def _extract_dual_thoughts_with_llm(thought_text):
    client = _get_llm_client()
    if client is None and not os.getenv(LLM_CONFIG.api_key_env):
        return None
    prompt = f"""
Read the rider thought below and extract:
1. Cs: bounded-rational intention
2. Cr: fully-rational intention

Return JSON only:
{{
  "Cs": "short phrase",
  "Cr": "short phrase"
}}

Thought:
{thought_text}
"""
    if client is not None:
        completion = client.chat.completions.create(
            model=os.getenv(LLM_CONFIG.model_env, LLM_CONFIG.model),
            messages=[
                {"role": "system", "content": "You extract dual intentions from rider thoughts and must return valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=LLM_CONFIG.temperature,
        )
        content = (completion.choices[0].message.content or "").strip()
    else:
        endpoint = os.getenv(LLM_CONFIG.base_url_env, LLM_CONFIG.default_base_url).rstrip("/") + "/chat/completions"
        payload = json.dumps(
            {
                "model": os.getenv(LLM_CONFIG.model_env, LLM_CONFIG.model),
                "messages": [
                    {"role": "system", "content": "You extract dual intentions from rider thoughts and must return valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": LLM_CONFIG.temperature,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv(LLM_CONFIG.api_key_env, '')}",
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = (result.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
    return _parse_json_like(content)


def _detect_emergent_intention_with_llm(cs, cr, thought_library):
    client = _get_llm_client()
    if client is None and not os.getenv(LLM_CONFIG.api_key_env):
        return None
    known_patterns = sorted(thought_library)[-20:]
    prompt = f"""
Decide whether the current rider intention is emergent relative to the known group thought library.
Return JSON only:
{{
  "emergent": true or false
}}

Current Cs: {cs}
Current Cr: {cr}
Known library: {known_patterns}
"""
    if client is not None:
        completion = client.chat.completions.create(
            model=os.getenv(LLM_CONFIG.model_env, LLM_CONFIG.model),
            messages=[
                {"role": "system", "content": "You judge whether a rider intention is emergent and must return valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=LLM_CONFIG.temperature,
        )
        content = (completion.choices[0].message.content or "").strip()
    else:
        endpoint = os.getenv(LLM_CONFIG.base_url_env, LLM_CONFIG.default_base_url).rstrip("/") + "/chat/completions"
        payload = json.dumps(
            {
                "model": os.getenv(LLM_CONFIG.model_env, LLM_CONFIG.model),
                "messages": [
                    {"role": "system", "content": "You judge whether a rider intention is emergent and must return valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": LLM_CONFIG.temperature,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv(LLM_CONFIG.api_key_env, '')}",
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = (result.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
    return _parse_json_like(content)


def extract_dual_thoughts(thought_text):
    r"""Extract bounded-rational and fully-rational intentions.

    对应机制公式 (1)
    LaTeX: (C_s, C_r)=f(\mathrm{thought})
    """

    text = str(thought_text or "").strip()
    if os.getenv(LLM_CONFIG.api_key_env):
        try:
            llm_result = _extract_dual_thoughts_with_llm(text)
            if isinstance(llm_result, dict):
                return {
                    "Cs": str(llm_result.get("Cs", "routine")),
                    "Cr": str(llm_result.get("Cr", "income")),
                }
        except Exception:
            _log_fallback_once()
    else:
        _log_fallback_once()
    lowered = text.lower()
    bounded_markers = ["anx", "worry", "tired", "risk", "fatigue", "competition", "stress", "焦虑", "风险"]
    rational_markers = ["income", "profit", "efficiency", "distance", "schedule", "收益", "效率", "距离"]
    cs = [marker for marker in bounded_markers if marker in lowered] or ["routine"]
    cr = [marker for marker in rational_markers if marker in lowered] or ["income"]
    return {"Cs": ", ".join(cs), "Cr": ", ".join(cr)}


def detect_emergent_intention(cs, cr, thought_library=None):
    r"""Detect whether an intention is emergent and update the group library.

    对应机制公式 (2)
    LaTeX: R_{t+1}=R_t \cup g(C_s, C_r)
    """

    thought_library = thought_library if thought_library is not None else set()
    if os.getenv(LLM_CONFIG.api_key_env):
        try:
            llm_result = _detect_emergent_intention_with_llm(cs, cr, thought_library)
            if isinstance(llm_result, dict):
                signature = f"{cs}|{cr}"
                is_emergent = bool(llm_result.get("emergent", False))
                if is_emergent:
                    thought_library.add(signature)
                return is_emergent, thought_library
        except Exception:
            _log_fallback_once()
    else:
        _log_fallback_once()
    signature = f"{cs}|{cr}"
    is_emergent = signature not in thought_library and cs != cr
    if is_emergent:
        thought_library.add(signature)
    return is_emergent, thought_library


def cosine_similarity(vector_a, vector_b):
    r"""Compute cosine similarity between two intention embeddings.

    对应机制公式 (3)
    LaTeX: \cos(\theta)=\frac{x\cdot y}{\lVert x\rVert \lVert y\rVert}
    """

    numerator = float(np.dot(vector_a, vector_b))
    denominator = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def cluster_intentions(texts, n_clusters=3):
    r"""Cluster intention embeddings with k-means.

    对应机制公式 (4)
    LaTeX: \mathcal{C}=\mathrm{kmeans}(\phi(\mathrm{text}))
    """

    if SentenceTransformer is None:
        print("Mechanism analysis fallback: sentence-transformers unavailable, using hashed text embeddings.")
        embeddings = []
        for text in texts:
            vector = np.zeros(32, dtype=float)
            for token in str(text).lower().split():
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                index = int(digest[:8], 16) % len(vector)
                vector[index] += 1.0
            norm = np.linalg.norm(vector)
            embeddings.append(vector / norm if norm > 0 else vector)
        embeddings = np.asarray(embeddings)
    else:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, show_progress_bar=False)
    kmeans = KMeans(n_clusters=min(n_clusters, len(texts)), random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return np.asarray(embeddings), labels


def load_thought_records(patterns=None):
    """Load thought records into a unified dataframe."""

    rows = []
    library = set()
    for path in discover_thought_files(patterns=patterns):
        source = path.parent.name
        rider_id = path.name.split("_")[0]
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, list):
            continue
        for record in payload:
            text = record.get("think") or record.get("mixed_thought") or ""
            dual = extract_dual_thoughts(text)
            is_emergent, library = detect_emergent_intention(dual["Cs"], dual["Cr"], library)
            result = record.get("result") or {}
            rows.append(
                {
                    "source": source,
                    "rider_id": rider_id,
                    "step": int(record.get("runner_step", 0)),
                    "thought": text,
                    "Cs": dual["Cs"],
                    "Cr": dual["Cr"],
                    "emergent": is_emergent,
                    "event_type": record.get("event_type", ""),
                    "selected_order_count": len(result.get("selected_order_ids", [])) if isinstance(result, dict) else 0,
                }
            )
    return pd.DataFrame(rows)


def assign_phase(step_series):
    """Map steps into three paper phases."""

    if step_series.empty:
        return []
    quantiles = step_series.quantile([1 / 3, 2 / 3]).tolist()

    def _map(value):
        if value <= quantiles[0]:
            return PHASE_LABELS[0]
        if value <= quantiles[1]:
            return PHASE_LABELS[1]
        return PHASE_LABELS[2]

    return step_series.map(_map)


def plot_cluster_scatter(df, embeddings, labels):
    """Generate Figure 7-12(a-c): clustered intention evolution."""

    projection = PCA(n_components=2).fit_transform(embeddings)
    df = df.copy()
    df["cluster"] = labels
    df["phase"] = assign_phase(df["step"])

    for phase in PHASE_LABELS:
        phase_df = df[df["phase"] == phase]
        if phase_df.empty:
            continue
        phase_projection = projection[phase_df.index]
        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(
            phase_projection[:, 0],
            phase_projection[:, 1],
            c=phase_df["cluster"],
            cmap="tab10",
            alpha=0.7,
        )
        ax.set_title(f"Figure 7-12 Cluster Scatter: {phase}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        unique_clusters = sorted(phase_df["cluster"].unique().tolist())
        handles = []
        for cluster_id in unique_clusters:
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=plt.cm.tab10(cluster_id % 10),
                    label=f"Cluster {cluster_id}",
                    markersize=8,
                )
            )
        ax.legend(handles=handles, title="Cluster")
        output_path = OUTPUT_DIR / f"fig7-12_cluster_scatter_{phase.lower().replace(' ', '_')}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_intention_behavior_heatmaps(df):
    """Generate Figure 7-12(d-f): intention-behavior correlation heatmaps."""

    df = df.copy()
    df["phase"] = assign_phase(df["step"])
    df["behavior_move"] = (df["event_type"] == "take_order").astype(int)
    df["behavior_orders"] = df["selected_order_count"]
    df["emergent_flag"] = df["emergent"].astype(int)

    for phase in PHASE_LABELS:
        phase_df = df[df["phase"] == phase]
        if phase_df.empty:
            continue
        indicators = pd.get_dummies(phase_df["Cs"].str.split(", ").str[0], prefix="intent")
        corr_df = pd.concat(
            [
                indicators,
                phase_df[["behavior_move", "behavior_orders", "emergent_flag"]].reset_index(drop=True),
            ],
            axis=1,
        )
        matrix = corr_df.corr().loc[indicators.columns, ["behavior_move", "behavior_orders", "emergent_flag"]]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(matrix, annot=True, cmap="Blues", ax=ax)
        ax.set_title(f"Figure 7-12 Correlation Heatmap: {phase}")
        output_path = OUTPUT_DIR / f"fig7-12_correlation_heatmap_{phase.lower().replace(' ', '_')}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def build_sankey(df):
    """Generate Figure 7-12(g): intention evolution Sankey diagram."""

    df = df.copy()
    df["phase"] = assign_phase(df["step"])
    dominant = (
        df.assign(primary_intent=df["Cs"].str.split(", ").str[0])
        .groupby(["source", "phase"])["primary_intent"]
        .agg(lambda values: Counter(values).most_common(1)[0][0])
        .reset_index()
    )

    phase_to_state = {
        PHASE_LABELS[0]: "Rule Following",
        PHASE_LABELS[1]: "Anxiety Driven",
        PHASE_LABELS[2]: "Risk Avoidance",
    }
    labels = list(phase_to_state.values())
    sources = [0, 1]
    targets = [1, 2]
    values = [max(1, len(dominant[dominant["phase"] == phase])) for phase in PHASE_LABELS[:2]]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=labels, pad=20, thickness=20),
                link=dict(source=sources, target=targets, value=values),
            )
        ]
    )
    fig.update_layout(title_text="Figure 7-12(g) Intention Evolution Sankey", font_size=12)
    output_path = OUTPUT_DIR / "fig7-12_intention_sankey.html"
    fig.write_html(str(output_path))
    return output_path


def plot_intervention_comparison(df):
    """Generate Figure 7-12(h-k): behavior transitions under different sources."""

    summary = (
        df.assign(primary_intent=df["Cs"].str.split(", ").str[0])
        .groupby(["source", "primary_intent"])["selected_order_count"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=summary, x="source", y="selected_order_count", hue="primary_intent", ax=ax)
    ax.set_title("Figure 7-12(h-k) Behavior Transition Comparison")
    ax.set_ylabel("Average Selected Order Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "fig7-12_intervention_behavior_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run mechanism analysis (Algorithm 7-3).")
    parser.add_argument(
        "--decision_mode",
        choices=["llm", "heuristic", "imitation", "auto"],
        default="llm",
        help="Reserved for pipeline consistency. Paper reproduction uses llm.",
    )
    parser.add_argument(
        "--thought_patterns",
        nargs="*",
        default=["simulation_results/sim_*/**/*_thought.json"],
        help="Glob patterns for thought files, relative to project root.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_thought_records(patterns=args.thought_patterns)
    if df.empty:
        raise SystemExit("未找到可用于机制分析的 *_thought.json 文件。")
    embeddings, labels = cluster_intentions(df["thought"].fillna("").tolist())
    df.to_csv(OUTPUT_DIR / "mechanism_dual_thoughts.csv", index=False)
    plot_cluster_scatter(df, embeddings, labels)
    plot_intention_behavior_heatmaps(df)
    build_sankey(df)
    plot_intervention_comparison(df)


if __name__ == "__main__":
    main()
