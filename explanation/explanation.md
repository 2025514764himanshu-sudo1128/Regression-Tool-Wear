# Experiment 07 — Code Explanation
# Regression Model for Tool Wear Predictive Maintenance

---

## What is this program doing?

In manufacturing, cutting tools wear out over time.
If a tool wears too much, it produces defective parts.
If you replace it too early, you waste money.

This program trains a Machine Learning model that:
1. Learns the relationship between cutting conditions and tool wear
2. Predicts how much wear will occur for new conditions
3. Estimates WHEN the tool will reach its failure limit
4. Helps schedule tool replacement BEFORE failure happens

This is called **Predictive Maintenance** — the future of manufacturing.

---

## Machine Learning Concepts Used

| Concept | Meaning |
|---|---|
| Supervised Learning | Model learns from labeled data (input → known output) |
| Linear Regression | Fits a straight line/plane through data |
| Training set | Data used to teach the model (80%) |
| Test set | Data used to evaluate the model (20%) |
| MSE | How wrong the predictions are on average |
| R² Score | How well the model explains the data |

---

## Line by Line Explanation

---

### Lines 1-5 (Imports)
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```
**Scikit-learn (sklearn):** The most widely used ML library in Python.
Contains all standard ML algorithms and tools.

**train_test_split:** Splits data into training and testing sets.

**LinearRegression:** The regression model we will train.

**mean_squared_error, r2_score:** Functions to measure how
accurate our model's predictions are.

---

### Lines 8-17 (Generating Dataset)
```python
data["Wear"] = (
    0.0005 * data["Speed"] +
    0.2    * data["Feed"]  +
    0.01   * data["Time"]  +
    np.random.normal(0, 0.01, 100)
)
```
**What is this relationship?**
Tool wear depends on:
- Higher cutting speed → more friction → more wear
- Higher feed rate → more material removed → more wear
- Longer time → more wear accumulated

**The random noise:**
`np.random.normal(0, 0.01, 100)` adds small random variations.
Real data always has noise — perfect relationships don't exist.

**This is our "ground truth" model.**
The regression will try to DISCOVER this relationship from data.

---

### Lines 20-21 (Features and Target)
```python
X = data[["Speed", "Feed", "Time"]]  # Input features
y = data["Wear"]                       # Target to predict
```
**X (Features/Inputs):** The variables we use to make predictions.
Also called independent variables.

**y (Target/Output):** What we want to predict.
Also called dependent variable.

**In our case:**
Given Speed, Feed, Time → predict Wear.

---

### Lines 24-27 (Train-Test Split)
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
**Why split data?**
We train on 80% and test on the remaining 20% that
the model has NEVER seen before.

This tells us if the model actually learned the pattern
or just memorized the training data (overfitting).

**test_size=0.2:**
20% of data goes to testing, 80% to training.
With 100 rows: 80 train, 20 test.

**random_state=42:**
Ensures the same split every time you run.
42 is a common convention (any number works).

**What are X_train, X_test, y_train, y_test?**
- X_train: Features for training (80 rows)
- y_train: Actual wear for training (80 values)
- X_test: Features for testing (20 rows)
- y_test: Actual wear for testing (20 values)

---

### Lines 30-32 (Model Training)
```python
model = LinearRegression()
model.fit(X_train, y_train)
```
**LinearRegression():**
Creates the model object — not trained yet, just initialized.

**model.fit(X_train, y_train):**
This is where the ACTUAL LEARNING happens!
The algorithm finds the best coefficients (a1, a2, a3, b) for:

`Wear = a1×Speed + a2×Feed + a3×Time + b`

It does this by minimizing the sum of squared errors
between predicted and actual wear values.

After fit(), the model knows the relationship in the data.

---

### Lines 35-36 (Predictions)
```python
y_pred = model.predict(X_test)
```
**model.predict():**
Uses the learned coefficients to predict wear for
the 20 test samples the model has never seen.

Returns an array of 20 predicted wear values.

---

### Lines 39-42 (Evaluation)
```python
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
```
**Mean Squared Error (MSE):**
Average of (actual - predicted)² for all test points.
Lower = better. 0 = perfect.

**R² Score (Coefficient of Determination):**
Measures what percentage of variation in wear is
explained by the model.

- R² = 1.0 → Perfect model (explains 100% of variation)
- R² = 0.9 → Excellent (explains 90%)
- R² = 0.0 → Useless model

**Expected results:**
Since we created the data with a known linear formula,
R² should be very high (≈ 0.98).

---

### Lines 45-56 (Predict Failure Time)
```python
time_values = np.linspace(1, 200, 1000)
wear_pred = model.predict(pd.DataFrame({
    "Speed": [speed] * len(time_values),
    "Feed":  [feed]  * len(time_values),
    "Time":  time_values
}))
critical_indices = np.where(wear_pred >= 0.3)[0]
critical_time = time_values[critical_indices[0]]
```
**np.linspace(1, 200, 1000):**
Creates 1000 evenly spaced time values from 1 to 200.
We will predict wear at each of these times.

**Fixing Speed and Feed:**
We want to know: at fixed operating conditions,
AT WHAT TIME does wear reach 0.3mm?

**np.where(wear_pred >= 0.3):**
Finds indices where predicted wear first reaches 0.3mm.
`[0]` gets the first occurrence.

**critical_time:**
The first time the tool reaches critical wear.
Replace the tool BEFORE this time!

---

## Visualizing Results
```python
plt.scatter(y_test, y_pred, ...)
plt.plot([min, max], [min, max], 'r--', ...)
```
**Actual vs Predicted plot:**
Each point = one test sample.
X-axis = actual wear, Y-axis = predicted wear.

**The red dashed line:**
This is the "perfect prediction" line (actual = predicted).
Points close to this line = good predictions.
Points far away = errors.

---

## Key ML Vocabulary

| Term | Meaning |
|---|---|
| Model | Mathematical function that maps input → output |
| Training | Finding the best function parameters from data |
| Prediction | Using trained model on new unseen data |
| Overfitting | Model memorizes training data but fails on new data |
| Underfitting | Model too simple to capture the pattern |
| Generalization | Model works well on data it hasn't seen |
