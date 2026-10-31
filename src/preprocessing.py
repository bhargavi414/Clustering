import csv
import math
import argparse
from pathlib import Path


def load_dataset(filename):
    """Load a numeric CSV dataset without a header."""

    data = []

    with open(filename, "r", newline="") as file:
        reader = csv.reader(file)

        for row_number, row in enumerate(reader, start=1):

            if not row or all(value.strip() == "" for value in row):
                continue

            values = []

            for col_number, value in enumerate(row, start=1):
                value = value.strip()

                if value == "":
                    values.append(None)
                    continue

                try:
                    number = float(value)

                    if not math.isfinite(number):
                        values.append(None)
                    else:
                        values.append(number)

                except ValueError:
                    raise ValueError(
                        f"Invalid value at row {row_number}, "
                        f"column {col_number}: {value}"
                    )

            data.append(values)

    if not data:
        raise ValueError("The dataset is empty.")

    dimensions = len(data[0])

    if dimensions == 0:
        raise ValueError("The dataset has no features.")

    for i, row in enumerate(data, start=1):
        if len(row) != dimensions:
            raise ValueError(
                f"Inconsistent number of features at row {i}."
            )

    return data


def handle_missing_values(data):
    """Replace missing values with the mean of their column."""

    rows = len(data)
    columns = len(data[0])

    column_means = []

    for j in range(columns):
        valid_values = [
            data[i][j]
            for i in range(rows)
            if data[i][j] is not None
        ]

        if not valid_values:
            raise ValueError(
                f"Column {j + 1} contains no valid values."
            )

        mean = sum(valid_values) / len(valid_values)
        column_means.append(mean)

    for i in range(rows):
        for j in range(columns):
            if data[i][j] is None:
                data[i][j] = column_means[j]

    return data


def standardize(data):
    """Apply Z-score standardization to every feature."""

    rows = len(data)
    columns = len(data[0])

    for j in range(columns):

        values = [data[i][j] for i in range(rows)]

        mean = sum(values) / rows

        variance = sum(
            (value - mean) ** 2 for value in values
        ) / rows

        std_dev = math.sqrt(variance)

        for i in range(rows):

            if std_dev == 0:
                data[i][j] = 0.0
            else:
                data[i][j] = (
                    data[i][j] - mean
                ) / std_dev

    return data


def save_dataset(data, filename):
    """Save processed data in numeric CSV format."""

    output_path = Path(filename)

    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(output_path, "w", newline="") as file:
        writer = csv.writer(file)

        for row in data:
            writer.writerow([
                f"{value:.10f}" for value in row
            ])


def preprocess_dataset(input_file, output_file,
                       apply_standardization=False):

    print(f"\nProcessing: {input_file}")

    data = load_dataset(input_file)

    print(f"Original data points: {len(data)}")
    print(f"Number of features: {len(data[0])}")

    data = handle_missing_values(data)

    print("Missing values handled.")

    if apply_standardization:
        data = standardize(data)
        print("Feature standardization completed.")

    save_dataset(data, output_file)

    print(f"Processed dataset saved to: {output_file}")


def main():

    parser = argparse.ArgumentParser(
        description="Preprocess datasets for K-Means clustering."
    )

    parser.add_argument(
        "input",
        help="Path to the input CSV dataset."
    )

    parser.add_argument(
        "output",
        help="Path to save the processed CSV dataset."
    )

    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Apply Z-score feature standardization."
    )

    args = parser.parse_args()

    try:
        preprocess_dataset(
            args.input,
            args.output,
            args.standardize
        )

    except (ValueError, OSError) as error:
        print(f"Preprocessing error: {error}")


if __name__ == "__main__":
    main()
