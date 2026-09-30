"""Task 2: preprocessing and feature analysis for student-mat.csv.

The preprocessor is deliberately fitted on X_train only. This prevents the
test set from influencing category levels, means, or standard deviations.
The transformed pandas DataFrames can be passed directly to a scikit-learn
regression model in Task 3.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "G3"


def load_student_math(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate the UCI Mathematics student dataset."""
    csv_path = Path(csv_path)
    data = pd.read_csv(csv_path, sep=";")

    if TARGET not in data.columns:
        raise ValueError(f"Required target column {TARGET!r} was not found.")
    if data.empty:
        raise ValueError("The dataset is empty.")
    if data.isna().any().any():
        missing = data.columns[data.isna().any()].tolist()
        raise ValueError(f"Missing values were found in: {missing}")

    return data


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the G3 target from the predictor columns."""
    X = data.drop(columns=TARGET).copy()
    y = data[TARGET].copy()
    return X, y


@dataclass
class TrainingPreprocessor:
    """Training-fitted standardization and one-hot encoding state."""

    numeric_features: list[str]
    categorical_features: list[str]
    means: pd.Series
    scales: pd.Series
    categories: dict[str, list[str]]
    output_columns: list[str]

    @classmethod
    def fit(cls, X_train: pd.DataFrame) -> "TrainingPreprocessor":
        """Learn preprocessing values from training features only."""
        if X_train.empty:
            raise ValueError("X_train is empty.")
        if X_train.isna().any().any():
            raise ValueError("X_train contains missing values.")

        numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
        categorical_features = [
            column for column in X_train.columns if column not in numeric_features
        ]

        means = X_train[numeric_features].mean()
        # ddof=0 matches the population standard deviation used by StandardScaler.
        scales = X_train[numeric_features].std(ddof=0).replace(0.0, 1.0)
        categories = {
            column: sorted(X_train[column].astype(str).unique().tolist())
            for column in categorical_features
        }

        dummy_columns = [
            f"{column}_{level}"
            for column in categorical_features
            # Omit one reference level to avoid redundant dummy columns in
            # Linear Regression models that include an intercept.
            for level in categories[column][1:]
        ]
        output_columns = numeric_features + dummy_columns

        return cls(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            means=means,
            scales=scales,
            categories=categories,
            output_columns=output_columns,
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the training-fitted transformation to any matching feature set."""
        expected = self.numeric_features + self.categorical_features
        missing_columns = [column for column in expected if column not in X.columns]
        if missing_columns:
            raise ValueError(f"Missing feature columns: {missing_columns}")
        if X[expected].isna().any().any():
            raise ValueError("Input features contain missing values.")

        numeric = (X[self.numeric_features] - self.means) / self.scales
        blocks: list[pd.DataFrame] = [numeric.astype(float)]

        for column in self.categorical_features:
            values = pd.Categorical(
                X[column].astype(str), categories=self.categories[column]
            )
            dummies = pd.get_dummies(values, prefix=column, dtype=float)
            dummies.index = X.index
            desired_columns = [
                f"{column}_{level}" for level in self.categories[column][1:]
            ]
            blocks.append(dummies.reindex(columns=desired_columns, fill_value=0.0))

        transformed = pd.concat(blocks, axis=1)
        return transformed.reindex(columns=self.output_columns, fill_value=0.0)

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Transform the same training data used to fit this instance."""
        return self.transform(X_train)

    def save_state(self, output_path: str | Path) -> None:
        """Save the fitted settings for documentation or later reuse."""
        state = {
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "means": self.means.to_dict(),
            "scales": self.scales.to_dict(),
            "categories": self.categories,
            "output_columns": self.output_columns,
        }
        Path(output_path).write_text(json.dumps(state, indent=2), encoding="utf-8")


def feature_correlation_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """Rank numeric and one-hot-encoded features by correlation with G3."""
    X, y = split_features_target(data)
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    # For binary dummy columns, Pearson correlation is also the point-biserial
    # correlation, so categorical levels can be compared with numeric features.
    encoded = pd.get_dummies(
        X,
        columns=categorical_features,
        drop_first=True,
        dtype=float,
    )
    correlations = encoded.corrwith(y).dropna()

    result = pd.DataFrame(
        {
            "feature": correlations.index,
            "correlation_with_G3": correlations.values,
        }
    )
    result["absolute_correlation"] = result["correlation_with_G3"].abs()
    return result.sort_values("absolute_correlation", ascending=False).reset_index(
        drop=True
    )


def write_task2_outputs(data: pd.DataFrame, output_dir: str | Path) -> None:
    """Write the actual feature-analysis results and a short explanation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, _ = split_features_target(data)
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
    correlations = feature_correlation_analysis(data)
    correlations.to_csv(output_dir / "feature_correlations.csv", index=False)

    top = correlations.head(10)
    top_lines = [
        f"{row.feature}: {row.correlation_with_G3:+.3f}"
        for row in top.itertuples(index=False)
    ]
    summary = [
        "Task 2 - Data Preprocessing and Feature Analysis",
        "",
        f"Rows: {len(data)}",
        f"Predictor columns before encoding: {X.shape[1]}",
        f"Numeric predictors standardized: {len(numeric_features)}",
        f"Categorical predictors one-hot encoded: {len(categorical_features)}",
        f"Missing values: {int(data.isna().sum().sum())}",
        f"Duplicate rows: {int(data.duplicated().sum())}",
        "",
        "Top absolute correlations with G3:",
        *top_lines,
        "",
        "Interpretation:",
        "G2 and G1 have the strongest positive relationships with the final grade.",
        "Past failures have the strongest negative relationship among the original",
        "numeric predictors. Correlation describes association, not causation.",
        "",
        "Preprocessing decisions:",
        "- G3 is separated before any feature transformation.",
        "- Text categories are one-hot encoded without assuming an order.",
        "- One reference level is omitted per category to prevent redundant",
        "  dummy columns in Linear Regression.",
        "- Numeric features are standardized using training-set statistics only.",
        "- Medu and Fedu remain numeric because the source already encodes their",
        "  ordered education levels from 0 to 4.",
        "- Unknown test-set categories produce zeros rather than new columns.",
        "- The same output column order is used for training and testing data.",
    ]
    (output_dir / "task2_results.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Task 2 feature analysis for student-mat.csv."
    )
    parser.add_argument("--data", default="student-mat.csv")
    parser.add_argument("--output-dir", default="task2_outputs")
    args = parser.parse_args()

    data = load_student_math(args.data)
    write_task2_outputs(data, args.output_dir)
    print(f"Task 2 outputs written to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
