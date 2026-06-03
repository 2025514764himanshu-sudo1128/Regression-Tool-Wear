# Experiment 07: Regression Model for Tool Wear Predictive Maintenance

**Subject:** AI in Mechanical Engineering (ONT406)
**Sharda University, Greater Noida**

---

## Aim
To develop a Linear Regression model using Scikit-learn to predict the remaining useful life (RUL) of a cutting tool.

---

## Concepts Covered
- Supervised Machine Learning
- Feature selection and train-test split
- Linear Regression model training
- Model evaluation using MSE and R² score
- Predicting tool failure time (Remaining Useful Life)

---

## Formulas Used

| Formula | Description |
|---|---|
| Wear = a1×Speed + a2×Feed + a3×Time + b | Linear Regression |
| MSE = (1/n)×Σ(actual-predicted)² | Mean Squared Error |
| R² = 1 - (SSres/SStot) | Coefficient of Determination |

---

## Software Required

| Software | Purpose | Download Link |
|---|---|---|
| Python 3.x | Programming language | https://www.python.org/downloads/ |
| VS Code | Code editor | https://code.visualstudio.com/ |
| Git | Version control | https://git-scm.com/ |

---

## Installation Steps

### Step 1: Install Python
```
1. Go to https://www.python.org/downloads/
2. Download Python 3.11 or above
3. CHECK "Add Python to PATH"
4. Verify: python --version
```

### Step 2: Install Required Libraries
```bash
pip install numpy pandas matplotlib scikit-learn
```

### Step 3: Verify Installation
```bash
python -c "import sklearn; print('Scikit-learn:', sklearn.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"
python -c "import matplotlib; print('Matplotlib:', matplotlib.__version__)"
```

---

## How to Run

```bash
git clone https://github.com/2025514764himanshu-sudo1128/Exp07-Regression-Tool-Wear.git
cd Exp07-Regression-Tool-Wear
python tool_wear_regression.py
```

---

## Output Files Generated
```
tool_wear_prediction.png   - Actual vs Predicted wear scatter plot
```

---

## Expected Console Output
```
Training samples: 80
Testing samples: 20
Model trained successfully!

Mean Squared Error (MSE): 0.000102
R² Score: 0.9823

Estimated Tool Failure Time: 23.45 minutes
Action: Replace tool before this time!
```

---

## Author
**Himanshu Kumar** (2025514764)
Department of Electrical, Electronics and Communication Engineering
Sharda University, Greater Noida
