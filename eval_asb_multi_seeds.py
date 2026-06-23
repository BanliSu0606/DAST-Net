import os
import re
import numpy as np
from scipy import stats
import pandas as pd
import glob
import argparse
from itertools import combinations

def extract_mpjpe_fde_from_log(log_content):
    results = {}
    
    # 支持多种格式
    patterns = {
        'MPJPE': [
            r'Total MPJPE:.*?mae:\s*([\d.]+)',
            r'MPJPE[:\s]+([\d.]+)',
            r'Average MPJPE[:\s]+([\d.]+)'
        ],
        'FDE': [
            r'Total FDE:.*?mae:\s*([\d.]+)',
            r'FDE[:\s]+([\d.]+)',
            r'Average FDE[:\s]+([\d.]+)'
        ]
    }
    
    for metric, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, log_content, re.IGNORECASE)
            if match:
                results[metric] = float(match.group(1))
                print(f"  Found {metric}: {results[metric]:.4f}")
                break
    
    return results

def extract_blc_from_log(log_content):
    results = {}
    
    patterns = {
        'mean_ble': r'Mean Bone Length Error:\s*([\d.]+)',
        'std_ble': r'Std\s*Bone Length Error:\s*([\d.]+)',
        'max_ble': r'Max\s*Bone Length Error:\s*([\d.]+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, log_content, re.IGNORECASE)
        if match:
            results[key] = float(match.group(1))
            print(f"  Found {key}: {results[key]:.6f}")
    
    return results


def extract_all_metrics_from_log(log_file_path):
    if not os.path.exists(log_file_path):
        print(f"Warning: Log file not found: {log_file_path}")
        return None
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    results = {}
    
    results.update(extract_mpjpe_fde_from_log(content))
    results.update(extract_blc_from_log(content))
    
    return results if results else None

def aggregate_metrics_from_logs(log_paths, seed_names=None):
    all_results = {}
    
    if seed_names is None:
        seed_names = [f"Seed_{i}" for i in range(len(log_paths))]
    
    for idx, log_path in enumerate(log_paths):
        seed_name = seed_names[idx] if idx < len(seed_names) else f"Seed_{idx}"
        print(f"\nProcessing {seed_name}: {log_path}")
        
        results = extract_all_metrics_from_log(log_path)
        if results:
            all_results[seed_name] = results
    
    return all_results

def compute_aggregated_statistics(all_results, metric_names=None):
    if metric_names is None:
        metric_names = set()
        for results in all_results.values():
            metric_names.update(results.keys())
        metric_names = sorted(metric_names)
    
    statistics = {}
    
    for metric in metric_names:
        values = []
        seed_names = []
        for seed, results in all_results.items():
            if metric in results:
                values.append(results[metric])
                seed_names.append(seed)
        
        if values:
            n = len(values)
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1) if n > 1 else 0
            sem_val = std_val / np.sqrt(n) if n > 0 else 0
            
            if n > 1:
                t_critical = stats.t.ppf(0.975, df=n-1)
                ci_lower = mean_val - t_critical * sem_val
                ci_upper = mean_val + t_critical * sem_val
            else:
                ci_lower, ci_upper = mean_val, mean_val
            
            median_val = np.median(values)
            q1, q3 = np.percentile(values, [25, 75]) if n > 1 else (mean_val, mean_val)
            
            statistics[metric] = {
                'seed_names': seed_names,
                'seed_values': values,
                'n': n,
                'mean': mean_val,
                'std': std_val,
                'sem': sem_val,
                'ci_95_lower': ci_lower,
                'ci_95_upper': ci_upper,
                'median': median_val,
                'q1': q1,
                'q3': q3,
                'min': np.min(values),
                'max': np.max(values),
                'range': np.max(values) - np.min(values)
            }
    
    return statistics

def perform_pairwise_t_tests(all_results, metric_names=None):
    if metric_names is None:
        metric_names = set()
        for results in all_results.values():
            metric_names.update(results.keys())
        metric_names = sorted(metric_names)
    
    model_metrics = {}
    for metric in metric_names:
        model_pattern = re.compile(r'(.+?)(?:_\d+)?$')
        model_name = model_pattern.match(metric)
        if model_name:
            base_name = model_name.group(1)
            if base_name not in model_metrics:
                model_metrics[base_name] = []
            model_metrics[base_name].append(metric)
    
    statistical_tests = {}
    
    for base_name, metrics in model_metrics.items():
        if len(metrics) >= 2:
            model_values = {}
            for metric in metrics:
                values = []
                for results in all_results.values():
                    if metric in results:
                        values.append(results[metric])
                if values:
                    model_values[metric] = values
            
            model_names = list(model_values.keys())
            for i, j in combinations(range(len(model_names)), 2):
                name1, name2 = model_names[i], model_names[j]
                vals1, vals2 = model_values[name1], model_values[name2]
                
                if len(vals1) == len(vals2) and len(vals1) > 1:
                    t_stat, p_value = stats.ttest_rel(vals1, vals2)
                    cohen_d = (np.mean(vals1) - np.mean(vals2)) / np.std(vals1 - vals2, ddof=1)
                    
                    statistical_tests[f"{name1}_vs_{name2}"] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'cohen_d': cohen_d,
                        'significant': p_value < 0.05,
                        'n': len(vals1)
                    }
    
    return statistical_tests

def format_comprehensive_report(statistics, statistical_tests=None):
    report = "\n" + "="*80 + "\n"
    report += "COMPREHENSIVE EVALUATION RESULTS\n"
    report += "="*80 + "\n"
    
    metric_categories = {
        'Position Errors': ['MPJPE', 'FDE'],
        'Bone Length Consistency': ['mean_ble', 'std_ble', 'max_ble'],
        'Dynamics Errors': ['rms_vel', 'rms_acc']
    }
    
    n_seeds = len(next(iter(statistics.values()))['seed_values']) if statistics else 0
    report += f"\nNumber of Seeds: {n_seeds}\n"
    report += f"Metric Categories: {len(metric_categories)}\n"
    
    for category, metrics in metric_categories.items():
        category_metrics = [m for m in metrics if m in statistics]
        if category_metrics:
            report += f"\n{category}:\n"
            report += "-"*40 + "\n"
            
            for metric in category_metrics:
                stats = statistics[metric]
                metric_name_display = metric.upper()
                
                report += f"\n  {metric_name_display} (n={stats['n']}):\n"
                report += f"    Individual: {', '.join([f'{v:.4f}' for v in stats['seed_values']])}\n"
                report += f"    Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}\n"
                report += f"    95% CI: [{stats['ci_95_lower']:.4f}, {stats['ci_95_upper']:.4f}]\n"
                report += f"    Median (Q1, Q3): {stats['median']:.4f} ({stats['q1']:.4f}, {stats['q3']:.4f})\n"
                report += f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n"
    
    if statistical_tests:
        report += "\n" + "="*80 + "\n"
        report += "STATISTICAL SIGNIFICANCE TESTS\n"
        report += "="*80 + "\n"
        
        for test_name, test_results in statistical_tests.items():
            report += f"\n{test_name}:\n"
            report += f"  t-statistic: {test_results['t_statistic']:.4f}\n"
            report += f"  p-value: {test_results['p_value']:.6f}\n"
            report += f"  Cohen's d: {test_results['cohen_d']:.4f}\n"
            report += f"  Significant at p<0.05: {'✓' if test_results['significant'] else '✗'}\n"
            report += f"  Sample size: {test_results['n']}\n"
    
    return report

def save_results_to_csv(statistics, output_file='aggregated_metrics.csv'):
    rows = []
    
    for metric, stats in statistics.items():
        for i, val in enumerate(stats['seed_values']):
            rows.append({
                'Metric': metric,
                'Seed': stats['seed_names'][i] if i < len(stats['seed_names']) else f'Seed_{i}',
                'Value': val,
                'Statistic_Type': 'Individual'
            })
        
        summary_stats = [
            ('Mean', stats['mean']),
            ('Std_Dev', stats['std']),
            ('SEM', stats['sem']),
            ('CI_95_Lower', stats['ci_95_lower']),
            ('CI_95_Upper', stats['ci_95_upper']),
            ('Median', stats['median']),
            ('Q1', stats['q1']),
            ('Q3', stats['q3']),
            ('Min', stats['min']),
            ('Max', stats['max']),
            ('Range', stats['range'])
        ]
        
        for stat_name, stat_value in summary_stats:
            rows.append({
                'Metric': metric,
                'Seed': 'Summary',
                'Value': stat_value,
                'Statistic_Type': stat_name
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    latex_file = output_file.replace('.csv', '_latex.tex')
    with open(latex_file, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\hline\n")
        f.write("Metric & Mean & Std & SEM & 95\\% CI & n \\\\\n")
        f.write("\\hline\n")
        
        for metric, stats in statistics.items():
            f.write(f"{metric.upper()} & ")
            f.write(f"{stats['mean']:.4f} & ")
            f.write(f"{stats['std']:.4f} & ")
            f.write(f"{stats['sem']:.4f} & ")
            f.write(f"[{stats['ci_95_lower']:.4f}, {stats['ci_95_upper']:.4f}] & ")
            f.write(f"{stats['n']} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Evaluation metrics across multiple seeds.}\n")
        f.write("\\label{tab:metrics}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to {latex_file}")

def find_log_files(base_dirs, log_pattern='log_eval.txt'):
    log_paths = []
    
    for base_dir in base_dirs:
        if '*' in base_dir:
            matched_dirs = glob.glob(base_dir)
            for matched_dir in matched_dirs:
                log_path = os.path.join(matched_dir, log_pattern)
                if os.path.exists(log_path):
                    log_paths.append(log_path)
        else:
            log_path = os.path.join(base_dir, log_pattern)
            if os.path.exists(log_path):
                log_paths.append(log_path)
            else:
                search_pattern = os.path.join(base_dir, '**', log_pattern)
                found_files = glob.glob(search_pattern, recursive=True)
                if found_files:
                    log_paths.extend(found_files)
                else:
                    print(f"Warning: No log file found in {base_dir}")
    
    return sorted(log_paths)

def main():
    parser = argparse.ArgumentParser(description='Extract and aggregate metrics from log files')
    parser.add_argument('--base_dirs', type=str, nargs='+',
                       help='Base directories containing log files (supports wildcards)')
    parser.add_argument('--log_pattern', type=str, default='log_eval.txt',
                       help='Log file pattern')
    parser.add_argument('--output', type=str, default='aggregated_metrics.csv',
                       help='Output CSV file name')
    parser.add_argument('--seed_names', type=str, nargs='+',
                       help='Custom seed names (optional)')
    args = parser.parse_args()
    
    # 如果没有指定base_dirs，使用默认值
    if not args.base_dirs:
        args.base_dirs = [
            'results/asb_st_0',
            'results/asb_st_1',
            'results/asb_st_2',
            'results/asb_st_3',
            'results/asb_st_4'
        ]
        print("Using default base directories")
    
    log_paths = find_log_files(args.base_dirs, args.log_pattern)
    
    if not log_paths:
        print("No log files found!")
        return
    
    print("="*80)
    print(f"Found {len(log_paths)} log files:")
    for path in log_paths:
        print(f"  {path}")
    
    all_results = aggregate_metrics_from_logs(log_paths, args.seed_names)
    
    if not all_results:
        print("No results extracted from logs!")
        return
    
    statistics = compute_aggregated_statistics(all_results)
    
    statistical_tests = perform_pairwise_t_tests(all_results)

    report = format_comprehensive_report(statistics, statistical_tests)
    print(report)
    
    save_results_to_csv(statistics, args.output)

    with open('evaluation_report.txt', 'w') as f:
        f.write(report)
    print("Report saved to evaluation_report.txt")
    
    if statistical_tests:
        print("\n" + "="*80)
        print("STATISTICAL TESTS SUMMARY")
        print("="*80)
        for test_name, test_results in statistical_tests.items():
            significance = "SIGNIFICANT" if test_results['significant'] else "not significant"
            print(f"{test_name}: t={test_results['t_statistic']:.4f}, p={test_results['p_value']:.6f} ({significance})")

if __name__ == '__main__':
    main()