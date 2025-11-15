"""
Compare DASP vs. R Regression Results
======================================

This script loads results from both DASP and R regression tests,
compares metrics, and generates comprehensive comparison reports.

Outputs:
- Metric comparison tables (CSV)
- Statistical summaries
- Detailed comparison reports (Markdown)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from scipy import stats

# Paths
BASE_DIR = Path(__file__).parent
DASP_RESULTS_DIR = BASE_DIR / "results" / "dasp_regression"
R_RESULTS_DIR = BASE_DIR / "results" / "r_regression"
COMPARISON_DIR = BASE_DIR / "results" / "comparisons"

# Create comparison directory
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

def load_results(results_dir, dataset_name):
    """Load all model results for a dataset."""
    dataset_dir = results_dir / dataset_name

    if not dataset_dir.exists():
        print(f"Warning: Directory not found: {dataset_dir}")
        return {}

    # Load all JSON files
    all_results = {}

    json_files = list(dataset_dir.glob("*.json"))
    if not json_files:
        print(f"Warning: No JSON files found in {dataset_dir}")
        return {}

    for json_file in json_files:
        if json_file.name == "all_models_summary.json":
            with open(json_file, 'r') as f:
                all_results.update(json.load(f))

    return all_results

def compare_datasets(dataset_name, dasp_results, r_results):
    """Compare DASP vs. R results for a single dataset."""

    print("=" * 80)
    print(f"COMPARING: {dataset_name.upper()}")
    print("=" * 80)

    # Find common models
    dasp_models = set(dasp_results.keys())
    r_models = set(r_results.keys())
    common_models = dasp_models & r_models

    print(f"\nDASP models: {len(dasp_models)}")
    print(f"R models: {len(r_models)}")
    print(f"Common models: {len(common_models)}")

    if len(common_models) == 0:
        print("ERROR: No common models found!")
        print(f"  DASP keys (first 5): {list(dasp_models)[:5]}")
        print(f"  R keys (first 5): {list(r_models)[:5]}")
        return None

    # Build comparison dataframe
    comparison_data = []

    for model_name in sorted(common_models):
        dasp_res = dasp_results[model_name]
        r_res = r_results[model_name]

        # Calculate differences
        r2_diff = dasp_res['r2'] - r_res['r2']
        rmse_diff = dasp_res['rmse'] - r_res['rmse']
        mae_diff = dasp_res['mae'] - r_res['mae']

        # Calculate percent differences
        r2_pct = (r2_diff / abs(r_res['r2'])) * 100 if r_res['r2'] != 0 else 0
        rmse_pct = (rmse_diff / r_res['rmse']) * 100 if r_res['rmse'] != 0 else 0

        comparison_data.append({
            'Model': dasp_res['model'],
            'DASP_R2': dasp_res['r2'],
            'R_R2': r_res['r2'],
            'R2_Diff': r2_diff,
            'R2_Diff_Pct': r2_pct,
            'DASP_RMSE': dasp_res['rmse'],
            'R_RMSE': r_res['rmse'],
            'RMSE_Diff': rmse_diff,
            'RMSE_Diff_Pct': rmse_pct,
            'DASP_MAE': dasp_res['mae'],
            'R_MAE': r_res['mae'],
            'MAE_Diff': mae_diff
        })

    df = pd.DataFrame(comparison_data)

    # Print summary statistics
    print("\n" + "-" * 80)
    print("METRIC COMPARISON SUMMARY")
    print("-" * 80)

    print("\nR² Comparison:")
    print(f"  Mean DASP R²: {df['DASP_R2'].mean():.4f}")
    print(f"  Mean R R²: {df['R_R2'].mean():.4f}")
    print(f"  Mean difference: {df['R2_Diff'].mean():.4f} ({df['R2_Diff_Pct'].mean():.2f}%)")
    print(f"  Max difference: {df['R2_Diff'].abs().max():.4f}")
    print(f"  Std of differences: {df['R2_Diff'].std():.4f}")

    print("\nRMSE Comparison:")
    print(f"  Mean DASP RMSE: {df['DASP_RMSE'].mean():.4f}")
    print(f"  Mean R RMSE: {df['R_RMSE'].mean():.4f}")
    print(f"  Mean difference: {df['RMSE_Diff'].mean():.4f} ({df['RMSE_Diff_Pct'].mean():.2f}%)")
    print(f"  Max difference: {df['RMSE_Diff'].abs().max():.4f}")

    # Statistical tests
    print("\n" + "-" * 80)
    print("STATISTICAL TESTS")
    print("-" * 80)

    # Paired t-test for R²
    t_stat_r2, p_value_r2 = stats.ttest_rel(df['DASP_R2'], df['R_R2'])
    print(f"\nPaired t-test (R²):")
    print(f"  t-statistic: {t_stat_r2:.4f}")
    print(f"  p-value: {p_value_r2:.4f}")
    print(f"  Significant at α=0.05: {'Yes' if p_value_r2 < 0.05 else 'No'}")

    # Paired t-test for RMSE
    t_stat_rmse, p_value_rmse = stats.ttest_rel(df['DASP_RMSE'], df['R_RMSE'])
    print(f"\nPaired t-test (RMSE):")
    print(f"  t-statistic: {t_stat_rmse:.4f}")
    print(f"  p-value: {p_value_rmse:.4f}")
    print(f"  Significant at α=0.05: {'Yes' if p_value_rmse < 0.05 else 'No'}")

    # Correlation of metrics
    r2_corr = np.corrcoef(df['DASP_R2'], df['R_R2'])[0, 1]
    rmse_corr = np.corrcoef(df['DASP_RMSE'], df['R_RMSE'])[0, 1]

    print(f"\nCorrelation between DASP and R:")
    print(f"  R² correlation: {r2_corr:.4f}")
    print(f"  RMSE correlation: {rmse_corr:.4f}")

    # Identify models with large differences
    print("\n" + "-" * 80)
    print("MODELS WITH LARGEST DIFFERENCES")
    print("-" * 80)

    # Top 5 by R² difference
    top_r2_diff = df.nlargest(5, 'R2_Diff', keep='all')[['Model', 'DASP_R2', 'R_R2', 'R2_Diff']]
    print("\nTop 5 R² differences (DASP > R):")
    print(top_r2_diff.to_string(index=False))

    # Bottom 5 by R² difference
    bottom_r2_diff = df.nsmallest(5, 'R2_Diff', keep='all')[['Model', 'DASP_R2', 'R_R2', 'R2_Diff']]
    print("\nTop 5 R² differences (R > DASP):")
    print(bottom_r2_diff.to_string(index=False))

    # Save comparison table
    output_file = COMPARISON_DIR / f"{dataset_name.lower().replace(' ', '_')}_comparison.csv"
    df.to_csv(output_file, index=False)
    print(f"\nComparison table saved to: {output_file}")

    # Generate summary statistics
    summary_stats = {
        'dataset': dataset_name,
        'n_models': len(df),
        'r2': {
            'dasp_mean': float(df['DASP_R2'].mean()),
            'r_mean': float(df['R_R2'].mean()),
            'mean_diff': float(df['R2_Diff'].mean()),
            'mean_diff_pct': float(df['R2_Diff_Pct'].mean()),
            'max_diff': float(df['R2_Diff'].abs().max()),
            'std_diff': float(df['R2_Diff'].std()),
            'correlation': float(r2_corr),
            't_stat': float(t_stat_r2),
            'p_value': float(p_value_r2)
        },
        'rmse': {
            'dasp_mean': float(df['DASP_RMSE'].mean()),
            'r_mean': float(df['R_RMSE'].mean()),
            'mean_diff': float(df['RMSE_Diff'].mean()),
            'mean_diff_pct': float(df['RMSE_Diff_Pct'].mean()),
            'max_diff': float(df['RMSE_Diff'].abs().max()),
            'std_diff': float(df['RMSE_Diff'].std()),
            'correlation': float(rmse_corr),
            't_stat': float(t_stat_rmse),
            'p_value': float(p_value_rmse)
        }
    }

    return df, summary_stats

def generate_markdown_report(summary_stats_list):
    """Generate a comprehensive Markdown report."""

    report_lines = [
        "# DASP vs. R Regression Comparison Report",
        "",
        "**Date Generated:** " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "This report compares regression model performance between DASP (Python) and R implementations",
        "using identical train/test splits and hyperparameters.",
        "",
        "### Datasets Tested",
        ""
    ]

    for stats in summary_stats_list:
        report_lines.extend([
            f"**{stats['dataset']}:**",
            f"- Models tested: {stats['n_models']}",
            f"- Mean R² (DASP): {stats['r2']['dasp_mean']:.4f}",
            f"- Mean R² (R): {stats['r2']['r_mean']:.4f}",
            f"- Mean difference: {stats['r2']['mean_diff']:.4f} ({stats['r2']['mean_diff_pct']:.2f}%)",
            f"- Correlation: {stats['r2']['correlation']:.4f}",
            ""
        ])

    report_lines.extend([
        "---",
        "",
        "## Detailed Results by Dataset",
        ""
    ])

    for stats in summary_stats_list:
        report_lines.extend([
            f"### {stats['dataset']}",
            "",
            "#### R² Comparison",
            "",
            f"| Metric | DASP | R | Difference |",
            f"|--------|------|---|------------|",
            f"| Mean | {stats['r2']['dasp_mean']:.4f} | {stats['r2']['r_mean']:.4f} | {stats['r2']['mean_diff']:.4f} |",
            f"| Max Difference | | | {stats['r2']['max_diff']:.4f} |",
            f"| Std of Diff | | | {stats['r2']['std_diff']:.4f} |",
            f"| Correlation | | | {stats['r2']['correlation']:.4f} |",
            "",
            f"**Paired t-test:** t={stats['r2']['t_stat']:.4f}, p={stats['r2']['p_value']:.4f}",
            "",
            "#### RMSE Comparison",
            "",
            f"| Metric | DASP | R | Difference |",
            f"|--------|------|---|------------|",
            f"| Mean | {stats['rmse']['dasp_mean']:.4f} | {stats['rmse']['r_mean']:.4f} | {stats['rmse']['mean_diff']:.4f} |",
            f"| Max Difference | | | {stats['rmse']['max_diff']:.4f} |",
            f"| Std of Diff | | | {stats['rmse']['std_diff']:.4f} |",
            f"| Correlation | | | {stats['rmse']['correlation']:.4f} |",
            "",
            f"**Paired t-test:** t={stats['rmse']['t_stat']:.4f}, p={stats['rmse']['p_value']:.4f}",
            "",
            "---",
            ""
        ])

    report_lines.extend([
        "## Interpretation",
        "",
        "### Acceptance Criteria",
        "",
        "- **R² match:** Within ±2% for deterministic models (PLS, Ridge, Lasso)",
        "- **R² match:** Within ±5% for stochastic models (Random Forest, XGBoost)",
        "- **Correlation:** >0.99 for identical implementations",
        "- **Statistical significance:** p > 0.05 indicates no systematic difference",
        "",
        "### Conclusions",
        ""
    ])

    # Analyze results
    for stats in summary_stats_list:
        report_lines.append(f"**{stats['dataset']}:**")

        # R² analysis
        r2_diff_pct = abs(stats['r2']['mean_diff_pct'])
        if r2_diff_pct < 2:
            report_lines.append(f"- [PASS] R² difference ({r2_diff_pct:.2f}%) is within ±2% threshold")
        elif r2_diff_pct < 5:
            report_lines.append(f"- [WARN] R² difference ({r2_diff_pct:.2f}%) is within ±5% but exceeds ±2%")
        else:
            report_lines.append(f"- [FAIL] R² difference ({r2_diff_pct:.2f}%) exceeds ±5% threshold")

        # Correlation analysis
        if stats['r2']['correlation'] > 0.99:
            report_lines.append(f"- [PASS] Correlation ({stats['r2']['correlation']:.4f}) indicates strong agreement")
        elif stats['r2']['correlation'] > 0.95:
            report_lines.append(f"- [WARN] Correlation ({stats['r2']['correlation']:.4f}) is good but below 0.99")
        else:
            report_lines.append(f"- [FAIL] Correlation ({stats['r2']['correlation']:.4f}) indicates poor agreement")

        # Statistical significance
        if stats['r2']['p_value'] > 0.05:
            report_lines.append(f"- [PASS] No significant difference (p={stats['r2']['p_value']:.4f} > 0.05)")
        else:
            report_lines.append(f"- [WARN] Significant difference detected (p={stats['r2']['p_value']:.4f} < 0.05)")

        report_lines.append("")

    report_lines.extend([
        "---",
        "",
        "## Files Generated",
        "",
        "- `bone_collagen_comparison.csv` - Detailed model-by-model comparison",
        "- `d13c_comparison.csv` - Detailed model-by-model comparison",
        "- `comparison_summary.json` - Summary statistics",
        "- `comparison_report.md` - This report",
        "",
        "---",
        "",
        "*Generated by DASP Testing Validation Framework*"
    ])

    report = "\n".join(report_lines)

    # Save report with UTF-8 encoding to handle special characters
    report_path = COMPARISON_DIR / "comparison_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nMarkdown report saved to: {report_path}")

    return report

def main():
    """Main execution function."""
    print("=" * 80)
    print("DASP vs. R Regression Comparison")
    print("=" * 80)

    summary_stats_list = []

    # Compare Bone Collagen
    print("\n")
    dasp_bone = load_results(DASP_RESULTS_DIR, "bone_collagen")
    r_bone = load_results(R_RESULTS_DIR, "bone_collagen")

    if dasp_bone and r_bone:
        bone_df, bone_stats = compare_datasets("Bone Collagen", dasp_bone, r_bone)
        if bone_stats:
            summary_stats_list.append(bone_stats)

    # Compare Enamel d13C
    print("\n")
    dasp_d13c = load_results(DASP_RESULTS_DIR, "d13c")
    r_d13c = load_results(R_RESULTS_DIR, "d13c")

    if dasp_d13c and r_d13c:
        d13c_df, d13c_stats = compare_datasets("Enamel d13C", dasp_d13c, r_d13c)
        if d13c_stats:
            summary_stats_list.append(d13c_stats)

    # Generate comprehensive report
    if summary_stats_list:
        report = generate_markdown_report(summary_stats_list)

        # Save summary statistics
        summary_path = COMPARISON_DIR / "comparison_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_stats_list, f, indent=2)
        print(f"Summary statistics saved to: {summary_path}")

    print("\n" + "=" * 80)
    print("COMPLETE! Comparison analysis finished.")
    print("=" * 80)
    print(f"\nResults saved to: {COMPARISON_DIR}")
    print("\nGenerated files:")
    print("  - bone_collagen_comparison.csv")
    print("  - d13c_comparison.csv")
    print("  - comparison_summary.json")
    print("  - comparison_report.md")

if __name__ == "__main__":
    main()
