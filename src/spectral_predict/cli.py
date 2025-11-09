"""Command-line interface for Spectral Predict."""

import argparse
import sys
from pathlib import Path
import pandas as pd

from . import __version__
from .io import read_csv_spectra, read_reference_csv, align_xy
from .search import run_search
from .report import write_markdown_report
from .interactive import run_interactive_loading
from .interactive_gui import run_interactive_loading_gui


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Spectral Predict - Automated spectral analysis software",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CSV wide format
  spectral-predict --spectra data/spectra.csv --reference data/ref.csv --id-column sample_id --target "%N"

  # ASD directory (ASCII)
  spectral-predict --asd-dir data/asd_sig --reference data/ref.csv --id-column filename --target "ADF"
        """,
    )

    parser.add_argument("--version", action="version", version=f"Spectral Predict {__version__}")

    # Input mode (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--spectra", type=str, help="Path to CSV file with spectral data")
    input_group.add_argument("--asd-dir", type=str, help="Directory containing ASD files")

    # Required arguments
    parser.add_argument(
        "--reference", type=str, required=True, help="Path to reference CSV with target variables"
    )
    parser.add_argument(
        "--id-column",
        type=str,
        required=True,
        help="ID column name in reference CSV (e.g., sample_id, filename)",
    )
    parser.add_argument(
        "--target", type=str, required=True, help="Target variable name from reference CSV"
    )

    # Optional arguments
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument(
        "--lambda-penalty",
        type=float,
        default=0.15,
        help="Complexity penalty weight (default: 0.15)",
    )
    parser.add_argument(
        "--max-n-components",
        type=int,
        default=24,
        help="Maximum number of PLS components to test (default: 24)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Maximum iterations for MLP models (default: 500)",
    )
    parser.add_argument(
        "--outdir", type=str, default="outputs", help="Output directory (default: outputs)"
    )
    parser.add_argument(
        "--asd-reader",
        type=str,
        default="auto",
        choices=["auto", "python", "rs-prospectr", "rs-asdreader"],
        help="ASD reader method (default: auto)",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        default=False,
        help="Skip interactive loading phase",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        default=False,
        help="Use text-based interactive mode instead of GUI",
    )

    args = parser.parse_args()

    # Print banner
    print("=" * 60)
    print(f"Spectral Predict v{__version__}")
    print("=" * 60)
    print()

    try:
        # Read spectral data
        print("Loading spectral data...")
        if args.spectra:
            X = read_csv_spectra(args.spectra)
            print(f"  Loaded {X.shape[0]} spectra with {X.shape[1]} wavelengths from CSV")
        elif args.asd_dir:
            # Import here to avoid circular dependency
            from .io import read_asd_dir

            X = read_asd_dir(args.asd_dir, reader_mode=args.asd_reader)
            print(f"  Loaded {X.shape[0]} spectra with {X.shape[1]} wavelengths from ASD directory")

        # Read reference data
        print("Loading reference data...")
        ref = read_reference_csv(args.reference, args.id_column)
        print(f"  Loaded reference data with {len(ref)} samples")

        # Align data
        print("Aligning spectral data with reference...")
        X_aligned, y = align_xy(X, ref, args.id_column, args.target)
        print(f"  Aligned {len(y)} samples for target '{args.target}'")
        print()

        # Run interactive loading phase
        if not args.no_interactive:
            if not args.no_gui:
                # Use GUI version (default)
                interactive_results = run_interactive_loading_gui(
                    X_aligned, y, args.id_column, args.target
                )
            else:
                # Use text-based version
                interactive_results = run_interactive_loading(
                    X_aligned, y, args.id_column, args.target
                )
            # Update X_aligned with potentially processed data (e.g., converted to absorbance)
            X_aligned = interactive_results['X']
            y = interactive_results['y']

        # Determine task type (regression or classification)
        # Check if target is numeric and has many decimal values (continuous)
        is_numeric = pd.api.types.is_numeric_dtype(y)
        has_decimals = is_numeric and (y % 1 != 0).any()  # Has non-integer values

        if is_numeric and (y.nunique() >= 10 or has_decimals):
            task_type = "regression"
            print(f"Detected regression task (continuous target, {y.nunique()} unique values)")
        else:
            task_type = "classification"
            print(f"Detected classification task ({y.nunique()} classes)")
        print()

        # Run search
        print(f"Running model search with {args.folds}-fold CV...")
        print(f"Lambda penalty: {args.lambda_penalty}")
        print()

        df_ranked = run_search(
            X_aligned, y, task_type, folds=args.folds, lambda_penalty=args.lambda_penalty,
            max_n_components=args.max_n_components, max_iter=args.max_iter
        )

        # Save results
        out_dir = Path(args.outdir)
        out_dir.mkdir(parents=True, exist_ok=True)

        results_path = out_dir / "results.csv"
        df_ranked.to_csv(results_path, index=False)
        print()
        print(f"Results saved to: {results_path}")

        # Generate report
        report_dir = Path("reports")
        report_path = write_markdown_report(args.target, df_ranked, report_dir)
        print(f"Report saved to: {report_path}")
        print()

        # Print top 3
        print("=" * 60)
        print("Top 3 Models:")
        print("=" * 60)
        top3 = df_ranked.head(3)

        if task_type == "regression":
            # Use itertuples() instead of iterrows() for better performance
            for row in top3.itertuples(index=False):
                print(f"\nRank {row.Rank}: {row.Model} ({row.SubsetTag})")
                print(f"  Preprocess: {row.Preprocess}")
                print(f"  RMSE: {row.RMSE:.4f}, R²: {row.R2:.4f}")
                print(f"  Variables: {int(row.n_vars)}/{int(row.full_vars)}")
                print(f"  Score: {row.CompositeScore:.4f}")
        else:
            # Use itertuples() instead of iterrows() for better performance
            for row in top3.itertuples(index=False):
                print(f"\nRank {row.Rank}: {row.Model} ({row.SubsetTag})")
                print(f"  Preprocess: {row.Preprocess}")
                print(f"  Accuracy: {row.Accuracy:.4f}", end="")
                if pd.notna(row.ROC_AUC):
                    print(f", ROC AUC: {row.ROC_AUC:.4f}")
                else:
                    print()
                print(f"  Variables: {int(row.n_vars)}/{int(row.full_vars)}")
                print(f"  Score: {row.CompositeScore:.4f}")

        print()
        print("Done!")

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
