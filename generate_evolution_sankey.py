import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_results")
EXPERIMENT_DIRS = [f"exp{i}" for i in range(1, 6)]
PHASE_LABELS = ["初始阶段", "中期阶段", "末期阶段", "最终阶段"]
INTENTION_LABELS = ["稳健维持", "竞争扩张", "风险收缩"]
SCHEDULE_LABELS = ["前移扩张", "稳定班次", "后移收缩"]


@dataclass
class DecisionRecord:
    step: int
    think_text: str
    before_start: float
    before_end: float
    after_start: float
    after_end: float
    avg_rank: float

    @property
    def before_duration(self) -> float:
        return compute_duration(self.before_start, self.before_end)

    @property
    def after_duration(self) -> float:
        return compute_duration(self.after_start, self.after_end)

    @property
    def duration_delta(self) -> float:
        return self.after_duration - self.before_duration

    @property
    def start_delta(self) -> float:
        return self.after_start - self.before_start

    @property
    def end_delta(self) -> float:
        return self.after_end - self.before_end


def extract_rider_id(filename: str) -> Optional[int]:
    match = re.match(r"(\d+)_thought\.json$", filename)
    return int(match.group(1)) if match else None


def parse_hour(time_text: object) -> Optional[float]:
    if not time_text or not isinstance(time_text, str):
        return None

    match = re.match(r"^\s*(\d{1,2})(?::(\d{1,2}))?", time_text)
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    return hour + minute / 60.0


def parse_rank(rank_text: object) -> Optional[int]:
    if not rank_text or not isinstance(rank_text, str):
        return None

    match = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", rank_text)
    if not match:
        return None

    return int(match.group(1))


def compute_duration(start_hour: float, end_hour: float) -> float:
    duration = end_hour - start_hour
    return duration + 24.0 if duration < 0 else duration


def count_keyword_hits(text: str, keywords: Sequence[str]) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def extract_decision_record(record: Dict[str, object]) -> Optional[DecisionRecord]:
    step = record.get("runner_step")
    params = record.get("param_dict")
    result = record.get("result")
    if not isinstance(step, int) or not isinstance(params, dict) or not isinstance(result, dict):
        return None

    before_start = parse_hour(params.get("before_go_work_time"))
    before_end = parse_hour(params.get("before_get_off_work_time"))
    after_start = parse_hour(result.get("go_work_time"))
    after_end = parse_hour(result.get("get_off_work_time"))
    if None in (before_start, before_end, after_start, after_end):
        return None

    ranks = [
        parse_rank(params.get("dis_rank")),
        parse_rank(params.get("money_rank")),
        parse_rank(params.get("order_rank")),
    ]
    valid_ranks = [rank for rank in ranks if rank is not None]
    if not valid_ranks:
        return None

    return DecisionRecord(
        step=step,
        think_text=str(record.get("think", "") or ""),
        before_start=before_start,
        before_end=before_end,
        after_start=after_start,
        after_end=after_end,
        avg_rank=sum(valid_ranks) / len(valid_ranks),
    )


def classify_intention(decision: DecisionRecord) -> str:
    text = decision.think_text.lower()

    maintain_keywords = [
        "keep the same",
        "stick with",
        "maintain",
        "no need to change",
        "not change",
        "current shift",
        "same schedule",
        "routine",
        "stable",
        "consistency",
    ]
    expansion_keywords = [
        "start earlier",
        "ending later",
        "end later",
        "extend",
        "longer shift",
        "extra hour",
        "more orders",
        "earn more",
        "higher demand",
        "peak time",
        "peak times",
        "morning rush",
        "lunch peak",
        "dinner rush",
        "capture",
        "maximize",
        "improve rankings",
        "competitive",
    ]
    risk_keywords = [
        "rest",
        "fatigue",
        "burnout",
        "tired",
        "well-being",
        "balance",
        "sustainable",
        "avoid",
        "overwork",
        "safer",
        "shorter shift",
        "reduce",
        "recovery",
        "energy levels",
    ]

    maintain_score = count_keyword_hits(text, maintain_keywords)
    expansion_score = count_keyword_hits(text, expansion_keywords)
    risk_score = count_keyword_hits(text, risk_keywords)

    duration_delta = decision.duration_delta
    start_earlier = decision.start_delta <= -0.75
    start_later = decision.start_delta >= 0.75
    end_later = decision.end_delta >= 0.75
    end_earlier = decision.end_delta <= -0.75
    large_change = abs(duration_delta) >= 1.0 or start_earlier or start_later or end_later or end_earlier

    # 排名越靠后，越容易表现出补偿性扩张；排名越靠前，越容易表现为维持或收缩。
    top_performer = decision.avg_rank <= 20
    mid_performer = 20 < decision.avg_rank <= 50

    if risk_score >= expansion_score + 1 and (duration_delta <= -0.5 or start_later or end_earlier):
        return "风险收缩"

    if expansion_score >= max(1, maintain_score) and (duration_delta >= 0.5 or start_earlier or end_later):
        return "竞争扩张"

    if maintain_score > 0 and not large_change:
        return "稳健维持"

    if top_performer and not large_change:
        return "稳健维持"

    if top_performer and duration_delta < 0:
        return "风险收缩"

    if mid_performer and duration_delta > 0.5:
        return "竞争扩张"

    if decision.avg_rank > 50 and (duration_delta > 0 or start_earlier or end_later):
        return "竞争扩张"

    if duration_delta <= -1.0:
        return "风险收缩"

    if duration_delta >= 1.0:
        return "竞争扩张"

    return "稳健维持"


def classify_schedule_strategy(decision: DecisionRecord) -> str:
    duration_delta = decision.duration_delta
    start_earlier = decision.start_delta <= -0.75
    end_later = decision.end_delta >= 0.75
    start_later = decision.start_delta >= 0.75
    end_earlier = decision.end_delta <= -0.75

    if duration_delta >= 0.5 or start_earlier or end_later:
        return "前移扩张"
    if duration_delta <= -0.5 or start_later or end_earlier:
        return "后移收缩"
    return "稳定班次"


def assign_phase(step_value: int, min_step: int, max_step: int) -> str:
    if max_step == min_step:
        return PHASE_LABELS[-1]

    ratio = (step_value - min_step) / (max_step - min_step)
    if ratio < 0.25:
        return PHASE_LABELS[0]
    if ratio < 0.5:
        return PHASE_LABELS[1]
    if ratio < 0.75:
        return PHASE_LABELS[2]
    return PHASE_LABELS[3]


def build_phase_state_map(
    decisions: Sequence[DecisionRecord],
    classifier,
) -> Dict[str, str]:
    if not decisions:
        return {}

    steps = [item.step for item in decisions]
    min_step = min(steps)
    max_step = max(steps)
    phase_buckets: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

    for decision in decisions:
        phase = assign_phase(decision.step, min_step, max_step)
        phase_buckets[phase].append((decision.step, classifier(decision)))

    phase_state_map: Dict[str, str] = {}
    for phase, items in phase_buckets.items():
        items.sort(key=lambda item: item[0])
        phase_state_map[phase] = items[-1][1]

    return phase_state_map


def load_rider_phase_maps():
    intention_phase_maps = []
    schedule_phase_maps = []

    for exp_name in EXPERIMENT_DIRS:
        exp_dir = os.path.join(BASE_DIR, exp_name)
        if not os.path.isdir(exp_dir):
            continue

        for filename in os.listdir(exp_dir):
            rider_id = extract_rider_id(filename)
            if rider_id is None:
                continue

            file_path = os.path.join(exp_dir, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            if not isinstance(data, list):
                continue

            decisions = []
            for record in data:
                if isinstance(record, dict):
                    decision = extract_decision_record(record)
                    if decision is not None:
                        decisions.append(decision)

            if not decisions:
                continue

            intention_map = build_phase_state_map(decisions, classify_intention)
            schedule_map = build_phase_state_map(decisions, classify_schedule_strategy)

            if intention_map:
                intention_phase_maps.append(
                    {"experiment": exp_name, "rider_id": rider_id, "phases": intention_map}
                )
            if schedule_map:
                schedule_phase_maps.append(
                    {"experiment": exp_name, "rider_id": rider_id, "phases": schedule_map}
                )

    return intention_phase_maps, schedule_phase_maps


def build_sankey_components(
    phase_maps: Sequence[Dict[str, object]],
    ordered_states: Sequence[str],
    palette: Sequence[str],
):
    node_labels: List[str] = []
    node_colors: List[str] = []
    node_index: Dict[Tuple[str, str], int] = {}

    for phase in PHASE_LABELS:
        for state_idx, state in enumerate(ordered_states):
            label = f"{phase}<br>{state}"
            node_index[(phase, state)] = len(node_labels)
            node_labels.append(label)
            node_colors.append(palette[state_idx])

    transition_counter: Counter = Counter()
    for item in phase_maps:
        phases = item["phases"]
        for current_phase, next_phase in zip(PHASE_LABELS, PHASE_LABELS[1:]):
            current_state = phases.get(current_phase)
            next_state = phases.get(next_phase)
            if current_state and next_state:
                transition_counter[
                    (
                        node_index[(current_phase, current_state)],
                        node_index[(next_phase, next_state)],
                    )
                ] += 1

    sources, targets, values = [], [], []
    for (source, target), value in transition_counter.items():
        sources.append(source)
        targets.append(target)
        values.append(value)

    return node_labels, node_colors, sources, targets, values


def build_summary_text(phase_maps: Sequence[Dict[str, object]]) -> str:
    phase_summary = []
    for phase in PHASE_LABELS:
        counter = Counter()
        for item in phase_maps:
            state = item["phases"].get(phase)
            if state:
                counter[state] += 1
        if counter:
            state, count = counter.most_common(1)[0]
            phase_summary.append(f"{phase}: {state} ({count})")
    return "<br>".join(phase_summary)


def add_sankey_trace(
    fig: go.Figure,
    col: int,
    title: str,
    phase_maps: Sequence[Dict[str, object]],
    ordered_states: Sequence[str],
    palette: Sequence[str],
):
    node_labels, node_colors, sources, targets, values = build_sankey_components(
        phase_maps,
        ordered_states,
        palette,
    )
    if not values:
        return

    fig.add_trace(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=20,
                thickness=18,
                line=dict(color="rgba(80,80,80,0.4)", width=0.5),
                label=node_labels,
                color=node_colors,
                hovertemplate="%{label}<extra></extra>",
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color="rgba(90,90,90,0.18)",
                hovertemplate="%{source.label} -> %{target.label}<br>数量: %{value}<extra></extra>",
            ),
        ),
        row=1,
        col=col,
    )

    fig.add_annotation(
        x=0.22 if col == 1 else 0.78,
        y=-0.10,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        text=f"{title}主导状态<br>{build_summary_text(phase_maps)}",
        font=dict(size=11),
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    intention_phase_maps, schedule_phase_maps = load_rider_phase_maps()

    if not intention_phase_maps:
        raise RuntimeError("未找到可用于生成意图桑基图的工时决策记录。")

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "sankey"}, {"type": "sankey"}]],
        subplot_titles=("骑手核心意图演化", "骑手工时策略演化"),
        column_widths=[0.5, 0.5],
    )

    add_sankey_trace(
        fig,
        1,
        "核心意图",
        intention_phase_maps,
        INTENTION_LABELS,
        ["#4C78A8", "#F58518", "#54A24B"],
    )
    add_sankey_trace(
        fig,
        2,
        "工时策略",
        schedule_phase_maps,
        SCHEDULE_LABELS,
        ["#4C78A8", "#B8B0AC", "#E45756"],
    )

    fig.update_layout(
        title="基于真实工时决策记录的多阶段演化过程桑基图",
        font=dict(size=12),
        width=1400,
        height=760,
        margin=dict(l=30, r=30, t=70, b=130),
    )

    output_file = os.path.join(OUTPUT_DIR, "evolution_process_sankey.html")
    fig.write_html(output_file, include_plotlyjs="cdn")

    print(f"Saved Sankey diagram to: {output_file}")
    print(f"Intention riders: {len(intention_phase_maps)}")
    print(f"Schedule riders: {len(schedule_phase_maps)}")
    print(f"Intention states: {Counter(state for item in intention_phase_maps for state in item['phases'].values())}")


if __name__ == "__main__":
    main()
