import argparse
from pathlib import Path

import pandas as pd

from data_loading import load_irrigation_data
from evaluation import evaluate_models, plot_feature_importance, plot_predicted_vs_actual
from feature_engineering import add_features
from models import build_models
from preprocessing import build_preprocessor, split_data
from simulation import plot_water_usage_comparison, run_simulation, simulation_metrics_row
from utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Virentus: Intelligent Irrigation Optimization System")
    parser.add_argument("--data", type=str, default="../irrigation_prediction.csv", help="Path to irrigation CSV file")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save outputs")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    plots_dir = ensure_dir(output_dir / "plots")

    df, target_col = load_irrigation_data(Path(args.data))
    df = add_features(df)

    X_train, X_test, y_train, y_test = split_data(
        df=df, target_col=target_col, test_size=args.test_size, random_state=args.random_state
    )

    preprocessor, _ = build_preprocessor(X_train)
    models = build_models(preprocessor)

    for model in models.values():
        model.fit(X_train, y_train)

    metrics_df, predictions = evaluate_models(models, X_test, y_test)
    best_model_name = metrics_df.iloc[0]["model"]
    best_pred = predictions[best_model_name]

    plot_predicted_vs_actual(y_test, best_pred, plots_dir / "predicted_vs_actual.png")

    if "random_forest" in models:
        plot_feature_importance(
            rf_pipeline=models["random_forest"],
            X_columns=X_train.columns.tolist(),
            output_path=plots_dir / "feature_importance_random_forest.png",
        )

    fixed_amount = float(y_train.mean())
    simulation_result = run_simulation(y_true=y_test, y_pred=best_pred, fixed_amount=fixed_amount)
    plot_water_usage_comparison(simulation_result, plots_dir / "water_usage_comparison.png")

    simulation_row = simulation_metrics_row(simulation_result, str(best_model_name))
    merged_metrics = pd.concat([metrics_df, simulation_row], ignore_index=True)
    merged_metrics.to_csv(output_dir / "metrics_summary.csv", index=False)
    save_json(simulation_result, output_dir / "simulation_results.json")

    print(f"Best model: {best_model_name}")
    print(f"Metrics saved to: {output_dir / 'metrics_summary.csv'}")
    print(f"Simulation saved to: {output_dir / 'simulation_results.json'}")
    print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()

