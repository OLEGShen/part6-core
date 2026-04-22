import os
import csv
import argparse
import json

from config import LLM_CONFIG, SIMULATION_CONFIG, SIMULATION_RESULTS_DIR
from simulation.city import City
from simulation.llm_agent import LLMDecisionError


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = str(SIMULATION_RESULTS_DIR)


def run_single_simulation(
    run_id,
    rider_num,
    run_len,
    one_day,
    order_weight,
    seed,
    save_detail=True,
    decision_mode="llm",
    llm_model=None,
    llm_base_url=None,
    business_district_num=SIMULATION_CONFIG.business_district_num,
    prompt_complexity="medium",
):
    print(f"[Run {run_id}] Starting simulation: riders={rider_num}, steps={run_len}, seed={seed}")

    save_path = os.path.join(RESULTS_DIR, f"sim_{run_id}")
    if save_detail:
        os.makedirs(save_path, exist_ok=True)

    city = City(
        rider_num=rider_num,
        run_len=run_len,
        one_day=one_day,
        order_weight=order_weight,
        seed=seed,
        output_dir=save_path if save_detail else None,
        decision_mode=decision_mode,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        business_district_num=business_district_num,
        prompt_complexity=prompt_complexity,
    )

    for step in range(run_len):
        city.step()
        if step % 100 == 0:
            print(f"[Run {run_id}] Step {step}/{run_len}")

    results = city.collect_results()

    if save_detail:
        _save_rider_details(results['rider_records'], save_path)
        _save_platform_record(results['platform_record'], save_path)
        _save_run_config(
            {
                "run_id": run_id,
                "rider_num": rider_num,
                "run_len": run_len,
                "one_day": one_day,
                "order_weight": order_weight,
                "seed": seed,
                "decision_mode": decision_mode,
                "llm_model": llm_model or "",
                "llm_base_url": llm_base_url or "",
                "business_district_num": business_district_num,
                "prompt_complexity": prompt_complexity,
            },
            save_path,
        )
        _save_time_series(results["time_series"], save_path)

    print(f"[Run {run_id}] Completed. Rider utility: {results['platform_record']['utility']:.2f}")
    return results, save_path


def _save_rider_details(rider_records, save_path):
    rider_file = os.path.join(save_path, 'rider_summary.csv')
    with open(rider_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rider_id', 'money', 'labor', 'total_order',
                         'go_work_time', 'get_off_work_time',
                         'stability', 'robustness', 'inv', 'utility'])
        for rider_id, data in rider_records.items():
            writer.writerow([
                rider_id,
                data['money'],
                data['labor'],
                data['total_order'],
                data['go_work_time'],
                data['get_off_work_time'],
                data['stability'],
                data['robustness'],
                data['inv'],
                data['utility']
            ])


def _save_platform_record(platform_record, save_path):
    platform_file = os.path.join(save_path, 'platform_summary.csv')
    with open(platform_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key, value in platform_record.items():
            writer.writerow([key, value])


def _save_run_config(run_config, save_path):
    config_file = os.path.join(save_path, "run_config.csv")
    with open(config_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for key, value in run_config.items():
            writer.writerow([key, value])


def _save_time_series(time_series, save_path):
    series_file = os.path.join(save_path, "time_series.csv")
    with open(series_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "involution", "swf", "platform_profit", "active_riders"])
        for step, involution, swf, profit, active_riders in zip(
            time_series["step"],
            time_series["involution"],
            time_series["swf"],
            time_series["platform_profit"],
            time_series["active_riders"],
        ):
            writer.writerow([step, involution, swf, profit, active_riders])

    positions_file = os.path.join(save_path, "rider_positions.json")
    with open(positions_file, "w", encoding="utf-8") as f:
        json.dump(time_series["rider_positions"], f, ensure_ascii=False, indent=2)


def run_multiple_simulations(num_runs=SIMULATION_CONFIG.num_runs, rider_num=SIMULATION_CONFIG.rider_num,
                              run_len=SIMULATION_CONFIG.run_len, one_day=SIMULATION_CONFIG.one_day,
                              order_weight=SIMULATION_CONFIG.order_weight, seed_base=SIMULATION_CONFIG.seed_base, save_detail=True,
                              decision_mode="llm", llm_model=None, llm_base_url=None):
    print(f"{'='*60}")
    print(f"Running {num_runs} simulations with {rider_num} riders")
    print(f"{'='*60}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    for run_id in range(num_runs):
        seed = seed_base + run_id if seed_base is not None else None
        results, save_path = run_single_simulation(
            run_id=run_id,
            rider_num=rider_num,
            run_len=run_len,
            one_day=one_day,
            order_weight=order_weight,
            seed=seed,
            save_detail=save_detail,
            decision_mode=decision_mode,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            business_district_num=SIMULATION_CONFIG.business_district_num,
        )
        all_results.append({
            'run_id': run_id,
            'results': results,
            'save_path': save_path
        })

    _aggregate_results(all_results, num_runs, rider_num, run_len)
    return all_results


def _aggregate_results(all_results, num_runs, rider_num, run_len):
    agg_file = os.path.join(RESULTS_DIR, 'aggregated_results.csv')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(agg_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run_id', 'rider_id', 'money', 'labor', 'total_order',
                       'stability', 'robustness', 'inv', 'utility',
                       'platform_profit', 'platform_fairness', 'platform_variety',
                       'platform_entropy', 'platform_utility', 'platform_swf', 'platform_involution'])

        for run_data in all_results:
            run_id = run_data['run_id']
            results = run_data['results']
            platform = results['platform_record']

            for rider_id, rider_data in results['rider_records'].items():
                writer.writerow([
                    run_id,
                    rider_id,
                    rider_data['money'],
                    rider_data['labor'],
                    rider_data['total_order'],
                    rider_data['stability'],
                    rider_data['robustness'],
                    rider_data['inv'],
                    rider_data['utility'],
                    platform['profit'],
                    platform['fairness'],
                    platform['variety'],
                    platform['entropy_increase'],
                    platform['utility'],
                    platform['swf'],
                    platform['involution'],
                ])

    _generate_summary_stats(all_results)


def _generate_summary_stats(all_results):
    summary_file = os.path.join(RESULTS_DIR, 'summary_statistics.csv')

    rider_metrics = {'money': [], 'labor': [], 'total_order': [],
                     'stability': [], 'robustness': [], 'inv': [], 'utility': []}
    platform_metrics = {'profit': [], 'fairness': [], 'variety': [],
                       'entropy_increase': [], 'utility': [], 'swf': [], 'involution': []}

    for run_data in all_results:
        results = run_data['results']
        for rider_data in results['rider_records'].values():
            rider_metrics['money'].append(rider_data['money'])
            rider_metrics['labor'].append(rider_data['labor'])
            rider_metrics['total_order'].append(rider_data['total_order'])
            rider_metrics['stability'].append(rider_data['stability'])
            rider_metrics['robustness'].append(rider_data['robustness'])
            rider_metrics['inv'].append(rider_data['inv'])
            rider_metrics['utility'].append(rider_data['utility'])

        for key in platform_metrics:
            platform_metrics[key].append(results['platform_record'][key])

    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric_type', 'metric_name', 'mean', 'std', 'min', 'max'])

        for metric_name, values in rider_metrics.items():
            if values:
                writer.writerow([
                    'rider',
                    metric_name,
                    f"{sum(values)/len(values):.4f}",
                    f"{_std(values):.4f}",
                    f"{min(values):.4f}",
                    f"{max(values):.4f}"
                ])

        for metric_name, values in platform_metrics.items():
            if values:
                writer.writerow([
                    'platform',
                    metric_name,
                    f"{sum(values)/len(values):.4f}",
                    f"{_std(values):.4f}",
                    f"{min(values):.4f}",
                    f"{max(values):.4f}"
                ])

    print(f"\n{'='*60}")
    print("Aggregated Results Summary")
    print(f"{'='*60}")
    print(f"Rider metrics averaged across {len(all_results)} runs:")
    for metric_name, values in rider_metrics.items():
        if values:
            print(f"  {metric_name}: mean={sum(values)/len(values):.4f}, std={_std(values):.4f}")
    print(f"\nPlatform metrics:")
    for metric_name, values in platform_metrics.items():
        if values:
            print(f"  {metric_name}: mean={sum(values)/len(values):.4f}, std={_std(values):.4f}")
    print(f"\nResults saved to: {RESULTS_DIR}")


def _std(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def main():
    parser = argparse.ArgumentParser(description='Run Meituan Rider Simulation')
    parser.add_argument('--num_runs', type=int, default=SIMULATION_CONFIG.num_runs, help='Number of simulation runs')
    parser.add_argument('--rider_num', type=int, default=SIMULATION_CONFIG.rider_num, help='Number of riders')
    parser.add_argument('--run_len', type=int, default=SIMULATION_CONFIG.run_len, help='Number of simulation steps')
    parser.add_argument('--one_day', type=int, default=SIMULATION_CONFIG.one_day, help='Steps per day')
    parser.add_argument('--order_weight', type=float, default=SIMULATION_CONFIG.order_weight, help='Order generation weight')
    parser.add_argument('--seed_base', type=int, default=SIMULATION_CONFIG.seed_base, help='Base random seed')
    parser.add_argument('--no_detail', action='store_true', help='Skip saving detailed results')
    parser.add_argument(
        '--decision_mode',
        choices=['auto', 'llm', 'heuristic', 'imitation'],
        default='llm',
        help='Rider decision backend (paper reproduction uses llm): auto / llm / heuristic / imitation',
    )
    parser.add_argument('--llm_model', type=str, default=LLM_CONFIG.model, help='LLM model name')
    parser.add_argument('--llm_base_url', type=str, default=None, help='LLM API base URL')

    args = parser.parse_args()

    try:
        run_multiple_simulations(
            num_runs=args.num_runs,
            rider_num=args.rider_num,
            run_len=args.run_len,
            one_day=args.one_day,
            order_weight=args.order_weight,
            seed_base=args.seed_base,
            save_detail=not args.no_detail,
            decision_mode=args.decision_mode,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
        )
    except LLMDecisionError as exc:
        raise SystemExit(
            f"无法启动 LLM 骑手模拟：{exc}\n"
            "请安装 `openai` 包并设置 `DASHSCOPE_API_KEY`，"
            "论文复现请使用 `--decision_mode llm`；"
            "仅开发调试可使用 `--decision_mode heuristic`。"
        )


if __name__ == '__main__':
    main()
