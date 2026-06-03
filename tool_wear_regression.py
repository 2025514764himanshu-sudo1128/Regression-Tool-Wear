import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# ============================================================

class DatasetError(ValueError):
    """Raised for invalid dataset parameters."""
    pass

class ModelError(RuntimeError):
    """Raised when model training or prediction fails."""
    pass

class PlotError(RuntimeError):
    """Raised when plotting fails."""
    pass

# -------------------------------------------------------
# Input Helpers
# -------------------------------------------------------
def get_positive_float(prompt):
    while True:
        try:
            value = float(input(prompt))
        except ValueError:
            print("  Error: Enter a numeric value.")
            continue
        if value <= 0:
            print("  Error: Value must be greater than zero.")
            continue
        return value

def get_positive_int(prompt, minimum=10):
    while True:
        try:
            value = int(input(prompt))
        except ValueError:
            print("  Error: Enter a whole number.")
            continue
        if value < minimum:
            print(f"  Error: Value must be at least {minimum}.")
            continue
        return value

def get_float_in_range(prompt, low, high):
    while True:
        try:
            value = float(input(prompt))
        except ValueError:
            print("  Error: Enter a numeric value.")
            continue
        if not (low <= value <= high):
            print(f"  Error: Value must be between {low} and {high}.")
            continue
        return value

def get_range(label):
    """Get a min-max range with max > min validation."""
    while True:
        try:
            low  = float(input(f"  {label} min: "))
            high = float(input(f"  {label} max: "))
        except ValueError:
            print("  Error: Enter numeric values.")
            continue
        if low <= 0:
            print("  Error: Minimum must be positive.")
            continue
        if high <= low:
            print("  Error: Maximum must be greater than minimum.")
            continue
        return low, high

# -------------------------------------------------------
# Dataset Generation
# -------------------------------------------------------
def generate_dataset(n, speed_r, feed_r, time_r):
    """Generate synthetic tool wear dataset."""
    if n < 10:
        raise DatasetError("Need at least 10 samples.")
    for label, (lo, hi) in [("Speed", speed_r), ("Feed", feed_r), ("Time", time_r)]:
        if hi <= lo:
            raise DatasetError(f"{label}: max must be greater than min.")

    np.random.seed(0)
    data = pd.DataFrame({
        "Speed": np.random.uniform(*speed_r, n),
        "Feed":  np.random.uniform(*feed_r,  n),
        "Time":  np.random.uniform(*time_r,  n),
    })
    data["Wear"] = (
        0.0005 * data["Speed"] +
        0.2    * data["Feed"]  +
        0.01   * data["Time"]  +
        np.random.normal(0, 0.01, n)
    ).clip(lower=0)
    return data

# -------------------------------------------------------
# Model Training & Evaluation
# -------------------------------------------------------
def train_model(data, test_frac):
    """Train LinearRegression and return model + test data."""
    required_cols = {"Speed", "Feed", "Time", "Wear"}
    missing = required_cols - set(data.columns)
    if missing:
        raise DatasetError(f"Dataset missing columns: {missing}")
    if len(data) < 10:
        raise DatasetError("Need at least 10 rows to train.")
    if not (0.1 <= test_frac <= 0.5):
        raise DatasetError("Test fraction must be between 0.1 and 0.5.")

    X = data[["Speed", "Feed", "Time"]]
    y = data["Wear"]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_frac, random_state=42
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    except (ValueError, TypeError) as e:
        raise ModelError(f"Model training failed: {e}")

    try:
        mse = float(mean_squared_error(y_test, y_pred))
        r2  = float(r2_score(y_test, y_pred))
    except (ValueError, TypeError) as e:
        raise ModelError(f"Evaluation metrics failed: {e}")

    print(f"\n{'='*55}")
    print("  MODEL EVALUATION RESULTS")
    print(f"{'='*55}")
    print(f"  Training samples : {len(X_train)}")
    print(f"  Testing samples  : {len(X_test)}")
    print(f"  MSE              : {mse:.6f}")
    print(f"  R² Score         : {r2:.4f}")
    if r2 >= 0.95:
        print(f"  Model Quality    : Excellent ✓")
    elif r2 >= 0.85:
        print(f"  Model Quality    : Good ✓")
    elif r2 >= 0.70:
        print(f"  Model Quality    : Acceptable")
    else:
        print(f"  Model Quality    : Poor — consider more data ✗")
    print(f"\n  Coefficients:")
    for feat, coef in zip(["Speed", "Feed", "Time"], model.coef_):
        print(f"    {feat:<8}: {coef:.6f}")
    print(f"    Intercept: {model.intercept_:.6f}")
    print(f"{'='*55}")

    return model, X_test, y_test, y_pred

# -------------------------------------------------------
# Failure Prediction
# -------------------------------------------------------
def predict_failure_time(model, speed, feed, critical_wear):
    """Find the time at which wear reaches the critical limit."""
    if speed <= 0 or feed <= 0:
        raise ModelError("Speed and feed must be positive.")
    if critical_wear <= 0:
        raise ModelError("Critical wear must be positive.")

    time_values = np.linspace(0.1, 500, 5000)
    try:
        wear_pred = model.predict(pd.DataFrame({
            "Speed": [speed] * len(time_values),
            "Feed":  [feed]  * len(time_values),
            "Time":  time_values,
        }))
    except (ValueError, TypeError) as e:
        raise ModelError(f"Prediction failed: {e}")

    above = np.where(wear_pred >= critical_wear)[0]
    if len(above) > 0:
        return float(time_values[above[0]])
    return None

# -------------------------------------------------------
# Plotting
# -------------------------------------------------------
def plot_actual_vs_predicted(y_test, y_pred, filename="tool_wear_prediction.png"):
    """Plot actual vs predicted wear values."""
    try:
        plt.figure(figsize=(7, 5))
        plt.scatter(y_test, y_pred, color="steelblue", alpha=0.7,
                    edgecolors="navy", linewidth=0.5, label="Predictions")
        lims = [
            min(float(y_test.min()), float(y_pred.min())),
            max(float(y_test.max()), float(y_pred.max()))
        ]
        plt.plot(lims, lims, "r--", linewidth=2, label="Perfect Prediction")
        plt.xlabel("Actual Wear (mm)")
        plt.ylabel("Predicted Wear (mm)")
        plt.title("Tool Wear: Actual vs Predicted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"  ✓ Plot saved: {filename}")
    except (TypeError, ValueError) as e:
        raise PlotError(f"Plot generation failed: {e}")
    except OSError as e:
        raise PlotError(f"Could not save plot: {e}")

# -------------------------------------------------------
# Main Program
# -------------------------------------------------------
def main():
    print("=" * 55)
    print("   EXPERIMENT 07: Tool Wear Regression Model")
    print("   AI in Mechanical Engineering — ONT406")
    print("   Sharda University")
    print("=" * 55)

    model  = None
    y_test = None
    y_pred = None

    while True:
        print("\n--- MENU ---")
        print("1. Generate Dataset and Train Model")
        print("2. Predict Tool Failure Time")
        print("3. Plot Actual vs Predicted Wear")
        print("4. Exit")

        choice = input("\nEnter your choice (1/2/3/4): ").strip()

        if choice in ["2", "3"] and model is None:
            print("  Error: Train a model first (option 1).")
            continue

        if choice == "1":
            try:
                print("\n--- Dataset Parameters ---")
                n         = get_positive_int(
                    "  Number of samples (min 20)              : ", minimum=20)
                print("  Cutting Speed range (m/min):")
                speed_r   = get_range("  Speed")
                print("  Feed Rate range (mm/rev):")
                feed_r    = get_range("  Feed")
                print("  Cutting Time range (min):")
                time_r    = get_range("  Time")
                test_frac = get_float_in_range(
                    "\n  Test fraction 0.1–0.5 (e.g. 0.2): ", 0.1, 0.5)

                data = generate_dataset(n, speed_r, feed_r, time_r)
                print(f"\n  ✓ Dataset generated: {n} samples.")
                model, X_test, y_test, y_pred = train_model(data, test_frac)

            except DatasetError as e:
                print(f"  Dataset Error: {e}")
            except ModelError as e:
                print(f"  Model Error: {e}")

        elif choice == "2":
            try:
                print("\n--- Predict Failure Time ---")
                speed         = get_positive_float("  Cutting speed (m/min)   : ")
                feed          = get_positive_float("  Feed rate (mm/rev)      : ")
                critical_wear = get_positive_float("  Critical wear limit (mm): ")
                failure_time  = predict_failure_time(model, speed, feed, critical_wear)

                print(f"\n  {'='*45}")
                if failure_time is not None:
                    print(f"  Estimated Failure Time : {failure_time:.2f} minutes")
                    print(f"  Recommendation         : Replace before {failure_time:.0f} min!")
                else:
                    print(f"  Tool will NOT reach {critical_wear}mm wear")
                    print(f"  under these operating conditions.")
                print(f"  {'='*45}")

            except ModelError as e:
                print(f"  Prediction Error: {e}")

        elif choice == "3":
            try:
                plot_actual_vs_predicted(y_test, y_pred)
            except PlotError as e:
                print(f"  Plot Error: {e}")

        elif choice == "4":
            print("\nExiting. Goodbye!")
            break

        else:
            print("  Error: Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Program interrupted by user. Goodbye!")
