# House Price Prediction — Linear Regression

A supervised machine learning project that predicts residential house sale prices using Linear Regression, built for HomeVista Properties.

---

## Problem Statement

HomeVista Properties handles thousands of residential property sales every year and wants to automate its house pricing process. The goal is to build a regression model that can accurately predict the market price of a house based on its physical features, location, and condition.

---

## Dataset

| Property | Value |
|---|---|
| Source | HomeVista Properties (internal) |
| Total rows | 2,919 |
| Training rows | 1,460 (with SalePrice) |
| Test rows | 1,459 (no SalePrice) |
| Features | 13 columns |
| Target | `SalePrice` |

### Features used

| Feature | Type | Description |
|---|---|---|
| `MSSubClass` | Numerical | Type of dwelling |
| `MSZoning` | Categorical | Zoning classification |
| `LotArea` | Numerical | Lot size in sq. ft. |
| `LotConfig` | Categorical | Lot configuration |
| `BldgType` | Categorical | Type of dwelling |
| `OverallCond` | Numerical | Condition rating (1–10) |
| `YearBuilt` | Numerical | Construction year |
| `YearRemodAdd` | Numerical | Remodel year |
| `Exterior1st` | Categorical | Exterior covering |
| `BsmtFinSF2` | Numerical | Finished basement sq. ft. |
| `TotalBsmtSF` | Numerical | Total basement sq. ft. |

---

## Workflow

```
Load Data → EDA → Preprocessing → Feature Scaling → Model Training → Evaluation
```

### Steps followed

1. **EDA** — explored distributions, missing values, and feature correlations
2. **Preprocessing** — filled missing values (median/mode), encoded 4 categorical columns using `LabelEncoder`
3. **Feature Scaling** — applied `StandardScaler` to normalise all features
4. **Train/Test Split** — 80% training, 20% testing (`random_state=42`)
5. **Model Training** — fit `LinearRegression` on training data
6. **Evaluation** — measured MAE, RMSE, and R² on unseen test data

---

## Results

| Metric | Value |
|---|---|
| R² Score | 0.6122 |
| MAE | $35,451 |

The model explains **61.2%** of the variation in house prices.


---

## Visualisations

| Plot | Description |
|---|---|
| SalePrice distribution | Shows original vs log-transformed target |
| Correlation bar chart | Feature correlation with SalePrice |
| Actual vs Predicted | How close predictions are to real prices |
| Residual plot | Checks for patterns in model errors |
| Feature importance | Coefficient magnitude per feature |

---

## Clone the repository
git clone git@github.com:madhav-rana/homevista-house-price-prediction.git

cd house-price-prediction-linear-regression


## Requirements

```
pandas
numpy
matplotlib
scikit-learn
jupyter
```

---

## Possible Improvements

- Apply log transformation on `SalePrice` to handle skewness
- Use `OneHotEncoder` instead of `LabelEncoder` for nominal categories
- Try **Ridge** or **Lasso** regression to handle multicollinearity
- Try **Random Forest** or **Gradient Boosting** for non-linear relationships
- Add more features (square footage, neighbourhood, garage, etc.)

---

## Author

**Madhav Singh Rana**  
B.Tech CSE
