"""Quick test to verify ranking fix works correctly."""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'src')

from spectral_predict.scoring import compute_composite_score


def test_user_bug_scenario():
    """Reproduce the user's bug: R²=0.943 model should rank well, not #876."""
    print("=" * 70)
    print("TEST: User's Bug Scenario - R²=0.943 model ranking")
    print("=" * 70)

    # Simulate 876 models like user's dataset
    np.random.seed(42)
    n_models = 876

    data = {
        "Model": ["PLS"] * n_models,
        "RMSE": np.random.uniform(0.1, 0.5, n_models),
        "R2": np.random.uniform(0.6, 0.95, n_models),
        "n_vars": np.random.randint(10, 2000, n_models),
        "full_vars": [2151] * n_models,
        "LVs": np.random.randint(5, 20, n_models),
        "Params": ["{}"] * n_models,
        "Preprocess": ["raw"] * n_models,
        "Deriv": [0] * n_models,
        "Window": [0] * n_models,
        "Poly": [0] * n_models,
        "SubsetTag": ["full"] * n_models,
        "top_vars": [None] * n_models,
    }
    df = pd.DataFrame(data)

    # Add the user's best model (by R²)
    df.loc[500, "R2"] = 0.943
    df.loc[500, "RMSE"] = 0.10
    df.loc[500, "n_vars"] = 2000  # Using many variables

    # Add hundreds of slightly worse models with fewer variables
    for i in range(50, 150):
        df.loc[i, "R2"] = np.random.uniform(0.88, 0.92)
        df.loc[i, "RMSE"] = np.random.uniform(0.11, 0.15)
        df.loc[i, "n_vars"] = np.random.randint(20, 200)

    # With the FIX and penalty=2, best R² model should rank well
    result = compute_composite_score(df, "regression", variable_penalty=2, complexity_penalty=2)

    best_r2_rank = result.loc[500, "Rank"]
    best_r2_score = result.loc[500, "CompositeScore"]

    print(f"\nModel with R²=0.943, RMSE=0.10, n_vars=2000:")
    print(f"  Rank: {best_r2_rank} (out of {n_models})")
    print(f"  CompositeScore: {best_r2_score:.4f}")

    # Show top 10 models
    print(f"\nTop 10 Models:")
    top10 = result.head(10)[["Rank", "R2", "RMSE", "n_vars", "LVs", "CompositeScore"]]
    print(top10.to_string(index=False))

    if best_r2_rank <= 50:
        print(f"\n[PASS] Best R2 model ranked #{best_r2_rank} (top 50)")
        return True
    else:
        print(f"\n[FAIL] Best R2 model ranked #{best_r2_rank}, expected top 50")
        return False


def test_penalty_zero_performance_only():
    """At penalty=0, ranking should be based purely on R² performance."""
    print("\n" + "=" * 70)
    print("TEST: Penalty=0 should rank by performance only")
    print("=" * 70)

    df = pd.DataFrame({
        "Model": ["PLS", "PLS", "PLS"],
        "RMSE": [0.05, 0.15, 0.10],
        "R2": [0.99, 0.85, 0.90],
        "n_vars": [2000, 20, 100],  # Best model uses MOST variables
        "full_vars": [2151, 2151, 2151],
        "LVs": [20, 5, 10],
        "Params": ["{}", "{}", "{}"],
        "Preprocess": ["raw", "raw", "raw"],
        "Deriv": [0, 0, 0],
        "Window": [0, 0, 0],
        "Poly": [0, 0, 0],
        "SubsetTag": ["full", "full", "full"],
        "top_vars": [None, None, None],
    })

    result = compute_composite_score(df, "regression", variable_penalty=0, complexity_penalty=0)

    print("\nResults at penalty=0:")
    print(result[["Rank", "R2", "RMSE", "n_vars", "LVs", "CompositeScore"]].to_string(index=False))

    if result.loc[0, "Rank"] == 1:
        print("\n[PASS] Best performance (R2=0.99) ranked #1 despite using most variables")
        return True
    else:
        print(f"\n[FAIL] Best performance ranked #{result.loc[0, 'Rank']}, expected #1")
        return False


def test_penalty_scaling():
    """Verify quadratic penalty scaling."""
    print("\n" + "=" * 70)
    print("TEST: Quadratic penalty scaling")
    print("=" * 70)

    df = pd.DataFrame({
        "Model": ["PLS", "PLS"],
        "RMSE": [0.10, 0.10],
        "R2": [0.90, 0.90],
        "n_vars": [100, 2000],
        "full_vars": [2151, 2151],
        "LVs": [10, 10],
        "Params": ["{}", "{}"],
        "Preprocess": ["raw", "raw"],
        "Deriv": [0, 0],
        "Window": [0, 0],
        "Poly": [0, 0],
        "SubsetTag": ["full", "full"],
        "top_vars": [None, None],
    })

    # At penalty=2, impact should be small
    result_p2 = compute_composite_score(df, "regression", variable_penalty=2, complexity_penalty=0)
    score_diff_p2 = abs(result_p2.loc[1, "CompositeScore"] - result_p2.loc[0, "CompositeScore"])

    # At penalty=10, impact should be much larger
    result_p10 = compute_composite_score(df, "regression", variable_penalty=10, complexity_penalty=0)
    score_diff_p10 = abs(result_p10.loc[1, "CompositeScore"] - result_p10.loc[0, "CompositeScore"])

    ratio = score_diff_p10 / score_diff_p2 if score_diff_p2 > 0 else 0

    print(f"\nScore difference (2000 vars vs 100 vars):")
    print(f"  At penalty=2:  {score_diff_p2:.6f}")
    print(f"  At penalty=10: {score_diff_p10:.6f}")
    print(f"  Ratio: {ratio:.1f} (expected ~25 for quadratic scaling)")

    if 20 < ratio < 30:
        print("\n[PASS] Penalty scaling is quadratic (ratio ~25)")
        return True
    else:
        print(f"\n[FAIL] Penalty scaling ratio {ratio:.1f}, expected ~25")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RANKING FIX VERIFICATION")
    print("Testing quadratic penalty scaling fix for scoring system")
    print("=" * 70)

    results = []
    results.append(test_user_bug_scenario())
    results.append(test_penalty_zero_performance_only())
    results.append(test_penalty_scaling())

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("\n[SUCCESS] ALL TESTS PASSED! Ranking fix is working correctly.")
        return 0
    else:
        print(f"\n[FAILED] {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
