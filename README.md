# DSN_Hackathon

# 🛒 Product-Store Sales Prediction

## 📌 Project Overview

This project was developed as part of a machine learning hackathon focused on predicting the **total sales of a product at a specific store**.

The objective was to use available product and store characteristics to build a regression model capable of predicting `total_sales` for unseen observations.

Rather than moving directly to model training, the project followed a complete machine learning workflow:

> **Data Understanding → Data Cleaning → Exploratory Data Analysis → Feature Assessment → Preprocessing → Model Development → Cross-Validation → Model Comparison → Hyperparameter Tuning → Model Interpretation → Final Prediction → Submission**

The project emphasizes not only predictive performance, but also **data quality, appropriate preprocessing, model comparison, metric interpretation, and reproducibility**.

---

## 🎯 Problem Statement

The task was to predict:

```text
total_sales
```

for a product sold at a particular store.

The available information included characteristics such as:

* Product weight
* Fat content
* Shelf visibility
* Product category
* Product price
* Store age
* Store size
* Store location tier
* Store format

Since `total_sales` is a continuous numerical variable, this was formulated as a:

> **Supervised Machine Learning Regression Problem**

---

# 📂 Dataset

The dataset was provided as separate training and test datasets.

### Training Dataset

The training data contained both the predictor variables and the target:

```text
Features + total_sales
```

### Test Dataset

The test data contained the predictor variables but did not contain `total_sales`.

```text
Features only
```

The test dataset was therefore kept completely separate during model development and was only used after the final modelling decisions had been made.

---

# 📊 Dataset Features

| Feature               | Description                                               | Type        |
| --------------------- | --------------------------------------------------------- | ----------- |
| `id`                  | Unique row identifier                                     | Identifier  |
| `product_code`        | Product identifier                                        | Identifier  |
| `product_weight_kg`   | Weight of the product in kilograms                        | Numerical   |
| `fat_content`         | Product fat-content category                              | Categorical |
| `shelf_visibility`    | Proportion of store display area allocated to the product | Numerical   |
| `product_category`    | Product category                                          | Categorical |
| `product_price`       | Listed price of the product                               | Numerical   |
| `store_code`          | Store identifier                                          | Identifier  |
| `store_age_years`     | Number of years the store has operated                    | Numerical   |
| `store_size`          | Size of the store                                         | Categorical |
| `store_location_tier` | Store location classification                             | Categorical |
| `store_format`        | Format/type of store                                      | Categorical |
| `total_sales`         | Total sales of the product at the store                   | **Target**  |

---

# 🧹 Data Cleaning

Before modelling, the dataset was inspected for:

* Missing values
* Inconsistent categorical values
* Identifier variables
* Data types
* Distribution of numerical variables
* Distribution of categorical variables

Several data-quality issues were identified and addressed.

---

## 1. Inconsistent Product Category Values

The product category column contained inconsistent representations of category names.

For example, differences in capitalization or formatting could cause the same category to be interpreted as separate categories.

This is particularly problematic when categorical variables are later encoded because:

```text
Category A
category a
CATEGORY A
```

could become separate features even though they represent the same category.

### Solution

A standardization function was applied to the product-category values so that equivalent categories were represented consistently.

The cleaned feature was standardized as:

```text
Proper Product_Category
```

The purpose was to ensure that the categorical values had a consistent representation before encoding.

### Why this matters

Data standardization prevents the model from treating formatting differences as meaningful differences between categories.

---

## 2. Missing Store Size

The `store_size` feature contained missing observations.

Possible values included:

```text
Small
Medium
Large
```

Rather than assigning a size without evidence, missing values were replaced with:

```text
Unknown
```

### Why?

Assigning an arbitrary size could introduce information that was not actually present in the original dataset.

Using `Unknown` preserves the original meaning:

> The store-size information was unavailable for this observation.

This also allows the model to learn whether observations with unknown store sizes behave differently from observations with known store sizes.

---

## 3. Missing Product Weight

The `product_weight_kg` column contained missing values.

Instead of replacing every missing value with one global median, **group-wise median imputation** was used.

The process was:

```text
Identify the relevant group
        ↓
Collect available product weights in that group
        ↓
Calculate the group's median weight
        ↓
Use that median for missing observations
```

### Why group-wise median?

Products belonging to different groups may naturally have different typical weights.

A single global median could ignore these differences.

Group-wise imputation therefore preserves more of the structure already present in the dataset.

### Why median?

Median is less sensitive to extreme values than the mean.

This makes it a useful measure for replacing missing numerical observations when outliers may be present.

---

# 🆔 Identifier Handling

The following columns were excluded from the modelling features:

```text
id
product_code
store_code
```

The modelling dataset was created using:

```python
X = train.drop(
    columns=['total_sales', 'id', 'product_code', 'store_code']
)

y = train['total_sales']
```

### Why remove `id`?

`id` represents the observation rather than a meaningful business characteristic.

Including arbitrary identifiers can introduce patterns that have no meaningful relationship with sales.

### Why remove `product_code` and `store_code`?

These columns were treated as identifiers rather than descriptive numerical measurements.

The model therefore focused on meaningful product and store characteristics such as:

* price
* visibility
* weight
* category
* store size
* location
* store format
* store age

---

# 🔍 Exploratory Data Analysis

EDA was performed before modelling to understand the target variable, individual predictors, categorical differences, and possible interactions.

The analysis focused on answering questions rather than simply producing visualizations.

---

## Target Variable Analysis

The distribution of `total_sales` was examined using:

* Descriptive statistics
* Distribution plots
* Boxplots
* Skewness analysis

This helped identify the central tendency, spread, possible outliers, and general shape of the sales distribution.

---

## Numerical Feature Analysis

The following numerical variables were examined against `total_sales`:

```text
product_weight_kg
shelf_visibility
product_price
store_age_years
```

Scatterplots and correlation analysis were used to investigate whether these variables were associated with differences in sales.

The analysis focused on identifying patterns that the machine learning models could potentially learn.

Correlation was not interpreted as causation.

---

## Categorical Feature Analysis

The following categorical variables were examined:

```text
fat_content
Proper Product_Category
store_size
store_location_tier
store_format
```

Group-level sales statistics and visualizations were used to investigate whether sales differed across categories.

Examples included:

* Average sales by store format
* Average sales by store size
* Average sales by location tier
* Sales distributions across product categories

---

# 🔗 Interaction Analysis

Potential interactions between variables were also investigated.

Examples included:

* Product category × store format
* Store format × location tier
* Product category × location tier
* Product price × shelf visibility

The purpose was to investigate whether the relationship between one variable and sales might depend on another variable.

This was particularly relevant because tree-based models can capture nonlinear relationships and interactions.

---

# 🛠️ Feature Engineering

Feature engineering was considered as part of the modelling process.

However, additional features were not created simply to increase the number of variables.

After examining the available product and store information, no additional engineered feature was considered sufficiently justified to add to the final model.

Therefore, the project proceeded using the cleaned original features.

This kept the modelling process:

* interpretable
* defensible
* reproducible
* closely aligned with the information actually provided by the dataset

---

# ⚙️ Data Preprocessing

After cleaning and EDA, the features were prepared for machine learning.

### Numerical Features

```python
numerical_features = [
    'product_weight_kg',
    'shelf_visibility',
    'product_price',
    'store_age_years'
]
```

### Categorical Features

```python
categorical_features = [
    'fat_content',
    'Proper Product_Category',
    'store_size',
    'store_location_tier',
    'store_format'
]
```

---

## One-Hot Encoding

Categorical variables were transformed using:

```python
OneHotEncoder(handle_unknown='ignore')
```

For example:

```text
store_size
```

could become:

```text
store_size_Small
store_size_Medium
store_size_Large
store_size_Unknown
```

### Why One-Hot Encoding?

The categories are nominal rather than numerical quantities.

Assigning arbitrary numbers such as:

```text
Small = 1
Medium = 2
Large = 3
```

could incorrectly imply a mathematical relationship between the categories.

One-hot encoding avoids introducing such artificial ordering.

---

## Handling Unknown Categories

The encoder used:

```python
handle_unknown='ignore'
```

This ensures that if the test dataset contains a category that was not encountered during training, the prediction pipeline does not fail.

---

# 🔄 Preprocessing Pipeline

Preprocessing and modelling were combined using a `Pipeline`.

Conceptually:

```text
Raw Features
     ↓
Column Transformer
     ↓
Numerical Features
     +
Categorical Features
     ↓
One-Hot Encoding
     ↓
Machine Learning Model
     ↓
Prediction
```

This ensured that the same preprocessing logic was applied consistently during:

* Cross-validation
* Model training
* Hyperparameter tuning
* Final test prediction

It also helped prevent preprocessing leakage during cross-validation.

---

# 🤖 Machine Learning Models

Four regression algorithms were evaluated:

1. Linear Regression
2. Random Forest Regression
3. Gradient Boosting Regression
4. XGBoost Regression

The models were evaluated using **5-fold cross-validation**.

---

# 1️⃣ Linear Regression

Linear Regression was used as the baseline model.

The purpose was to establish how well a relatively simple model could perform before introducing more complex algorithms.

### Why Linear Regression?

It provides:

* A simple baseline
* Fast training
* Easy interpretation
* A reference point for more complex models

However, sales relationships may contain nonlinearities and interactions that a linear model cannot adequately capture.

### Results

| Metric |       Score |
| ------ | ----------: |
| MAE    |  **839.80** |
| RMSE   | **1129.73** |
| R²     |  **0.5565** |

The model explained approximately **55.65% of the variation in sales under cross-validation**.

This became the baseline against which subsequent models were compared.

---

# 2️⃣ Random Forest Regression

Random Forest was introduced to capture nonlinear relationships and interactions that Linear Regression may miss.

Random Forest builds multiple decision trees and combines their predictions.

### Why Random Forest?

It can:

* Model nonlinear relationships
* Capture interactions
* Handle mixed feature types after preprocessing
* Reduce the instability of individual decision trees through ensemble learning

### Results

| Metric | Linear Regression | Random Forest |
| ------ | ----------------: | ------------: |
| MAE ↓  |            839.80 |    **781.33** |
| RMSE ↓ |           1129.73 |   **1119.93** |
| R² ↑   |            0.5565 |    **0.5642** |

Random Forest improved upon the linear baseline across all three metrics.

---

# 3️⃣ Gradient Boosting Regression

Gradient Boosting was then evaluated.

Unlike Random Forest, where trees are built independently, Gradient Boosting builds trees sequentially.

Each new tree attempts to improve upon the errors made by previous trees.

### Why Gradient Boosting?

It is particularly effective at learning:

* Nonlinear relationships
* Complex interactions
* Patterns missed by simpler models

### Results

| Metric | Random Forest | Gradient Boosting |
| ------ | ------------: | ----------------: |
| MAE ↓  |        781.33 |        **765.43** |
| RMSE ↓ |       1119.93 |       **1089.83** |
| R² ↑   |        0.5642 |        **0.5873** |

Gradient Boosting produced a clear improvement over Random Forest.

---

# 4️⃣ XGBoost

XGBoost was introduced as an advanced gradient-boosting implementation.

It provides several mechanisms for controlling model complexity and improving predictive performance.

### Why XGBoost?

XGBoost was considered because:

* Gradient Boosting was already performing strongly.
* XGBoost is designed for high-performance gradient boosting.
* It supports regularization.
* It can model nonlinear relationships and interactions.
* It provides useful controls for tuning model complexity.

### Initial Configuration

```python
XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
    objective='reg:squarederror',
    n_jobs=-1
)
```

### Results

| Model             |      MAE ↓ |      RMSE ↓ |       R² ↑ |
| ----------------- | ---------: | ----------: | ---------: |
| Linear Regression |     839.80 |     1129.73 |     0.5565 |
| Random Forest     |     781.33 |     1119.93 |     0.5642 |
| Gradient Boosting |     765.43 |     1089.83 |     0.5873 |
| **XGBoost**       | **762.72** | **1088.40** | **0.5884** |

XGBoost achieved the strongest baseline result across all three metrics.

However, the difference between XGBoost and Gradient Boosting was relatively small.

### XGBoost vs Gradient Boosting

```text
MAE:  765.43 → 762.72
RMSE: 1089.83 → 1088.40
R²:   0.5873 → 0.5884
```

Therefore, XGBoost was selected for further optimization, but the results did not indicate a dramatic difference between the two boosting approaches.

---

# 📏 Model Evaluation Metrics

Three metrics were used.

## Mean Absolute Error — MAE

MAE measures the average absolute difference between actual and predicted values.

```text
Lower = Better
```

For example, an MAE of approximately `762.72` means the model's absolute prediction error was approximately 763 sales units on average under the cross-validation evaluation.

---

## Root Mean Squared Error — RMSE

RMSE gives greater weight to larger prediction errors.

```text
Lower = Better
```

Because errors are squared before averaging, large prediction errors have a greater effect on RMSE than on MAE.

The difference between MAE and RMSE can therefore provide an indication of the presence of larger errors.

---

## R² — Coefficient of Determination

R² measures the proportion of variation in the target explained by the model relative to a mean-prediction baseline.

```text
Higher = Better
```

The initial XGBoost model achieved:

```text
R² = 0.5884
```

This means approximately **58.84% of the variation in `total_sales` was explained by the model under the cross-validation evaluation**.

R² should not be interpreted as prediction accuracy.

---

# 🔀 Why Cross-Validation?

The competition already provided separate training and test datasets, so an additional permanent train/test split was not required.

Instead, **5-fold cross-validation** was performed using only the training data.

The process was:

```text
Training Data
     ↓
 ┌───┬───┬───┬───┬───┐
 │ F1│ F2│ F3│ F4│ F5│
 └───┴───┴───┴───┴───┘
     ↓
Each fold becomes validation data once
     ↓
Average performance
```

This allowed the models to be evaluated across different portions of the training dataset while keeping the competition test set completely untouched.

---

# 🎯 Hyperparameter Tuning

After XGBoost produced the strongest baseline performance, hyperparameter tuning was performed.

The objective was to determine whether better XGBoost settings could improve performance beyond the initial configuration.

`RandomizedSearchCV` was used.

---

## Parameters Tuned

The search included:

```text
n_estimators
learning_rate
max_depth
min_child_weight
subsample
colsample_bytree
reg_alpha
reg_lambda
```

These parameters control:

* Number of trees
* Learning rate
* Tree complexity
* Minimum child weight
* Row sampling
* Feature sampling
* L1 regularization
* L2 regularization

---

## Search Strategy

Thirty different parameter combinations were randomly sampled.

Each combination was evaluated using 5-fold cross-validation.

Therefore:

```text
30 combinations × 5 folds
= 150 model fits
```

The search was optimized using:

```python
scoring='neg_mean_absolute_error'
```

MAE was selected as the primary tuning metric because it provides a direct interpretation of the average absolute prediction error.

---

# 🧠 Why Hyperparameter Tuning Matters

The initial XGBoost model used:

```text
n_estimators = 300
learning_rate = 0.05
max_depth = 3
```

These were initial settings rather than guaranteed optimal values.

Hyperparameter tuning allowed the model to systematically explore alternative configurations.

The process can be summarized as:

```text
Initial XGBoost
      ↓
Define reasonable parameter ranges
      ↓
Randomly sample combinations
      ↓
5-fold cross-validation
      ↓
Compare MAE
      ↓
Select best configuration
      ↓
Evaluate tuned model
```

Tuning was not assumed to automatically improve performance. The tuned model was compared against the original XGBoost results.

---

# 📊 Final Model Comparison

The baseline model comparison was:

| Model             |           MAE ↓ |          RMSE ↓ |            R² ↑ |
| ----------------- | --------------: | --------------: | --------------: |
| Linear Regression |          839.80 |         1129.73 |          0.5565 |
| Random Forest     |          781.33 |         1119.93 |          0.5642 |
| Gradient Boosting |          765.43 |         1089.83 |          0.5873 |
| XGBoost           |      **762.72** |     **1088.40** |      **0.5884** |
| Tuned XGBoost     | *See final run* | *See final run* | *See final run* |

> **Note:** The tuned XGBoost values should be updated with the actual results from the final tuning run rather than using illustrative values.

---

# 🔎 Feature Importance

Feature importance was extracted from the final XGBoost model to understand which processed features were most influential in the model's predictions.

Because categorical variables were one-hot encoded, the model sees encoded features such as:

```text
store_format_Superstore
store_size_Large
store_location_tier_Tier_1
```

rather than the original categorical column alone.

Feature importance was therefore examined after preprocessing.

### Important interpretation

Feature importance indicates how useful a feature was to the trained model.

It does **not** prove that the feature causes an increase or decrease in sales.

Therefore, the results are interpreted as:

> **Features that the model relied on most strongly for prediction**

rather than causal drivers of sales.

---

# 🏗️ End-to-End Architecture

```text
                 ┌──────────────────────┐
                 │    Raw Train Data    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Data Understanding   │
                 │ & Quality Assessment │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    Data Cleaning     │
                 │                      │
                 │ • Category cleanup   │
                 │ • Missing store size │
                 │ • Group median weight│
                 │ • Remove identifiers │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │         EDA          │
                 │                      │
                 │ • Target analysis    │
                 │ • Numerical analysis │
                 │ • Categorical        │
                 │ • Interactions       │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    Feature Review    │
                 │                      │
                 │ No unjustified       │
                 │ engineered features  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    Preprocessing     │
                 │                      │
                 │ • One-hot encoding   │
                 │ • Pipeline            │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  Model Development   │
                 └──────────┬───────────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
   Linear Regression   Random Forest   Gradient Boosting
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
                       XGBoost
                            │
                            ▼
                    5-Fold Cross-
                       Validation
                            │
                            ▼
                    Model Comparison
                            │
                            ▼
                     XGBoost Selected
                            │
                            ▼
                 Hyperparameter Tuning
                            │
                            ▼
                    Tuned XGBoost
                            │
                            ▼
                   Feature Importance
                            │
                            ▼
                  Final Model Training
                            │
                            ▼
                     Unseen Test Data
                            │
                            ▼
                   Same Preprocessing
                            │
                            ▼
                       Predictions
                            │
                            ▼
                     submission.csv
```

---

# 🚀 Final Prediction Workflow

After model selection and tuning:

### Step 1 — Train final model

The selected model was fitted using the complete training dataset.

### Step 2 — Prepare test data

The test data was transformed using the same feature structure and preprocessing logic.

### Step 3 — Generate predictions

```python
test_predictions = best_xgb_model.predict(X_test)
```

### Step 4 — Create submission

```python
submission = pd.DataFrame({
    'id': test_ids,
    'total_sales': test_predictions
})
```

### Step 5 — Export

```python
submission.to_csv(
    'submission.csv',
    index=False
)
```

---

# 🧪 Reproducibility

Random seeds were specified where appropriate, for example:

```python
random_state=42
```

This helps make model training and randomized search reproducible.

The project also used pipelines to ensure that preprocessing and modelling remained connected throughout training and prediction.

---

# 🧰 Technologies Used

* **Python**
* **Pandas** — data manipulation and cleaning
* **NumPy** — numerical operations
* **Matplotlib** — visualization
* **Scikit-learn** — preprocessing, cross-validation, evaluation, and machine learning
* **XGBoost** — gradient-boosted regression
* **SciPy** — randomized hyperparameter distributions
* **Jupyter Notebook** — experimentation and analysis

---

# 📌 Key Findings

1. The dataset required careful cleaning before modelling.
2. Inconsistent product-category values were standardized to prevent duplicate categorical representations.
3. Missing store sizes were represented as `Unknown` rather than being assigned an unsupported category.
4. Missing product weights were imputed using group-wise medians to preserve group-level differences.
5. Identifier columns were excluded from the modelling features.
6. EDA showed that sales relationships needed to be investigated beyond simple linear relationships.
7. Tree-based models outperformed the Linear Regression baseline.
8. Gradient Boosting significantly improved upon Random Forest.
9. XGBoost produced the strongest baseline performance.
10. The improvement from Gradient Boosting to XGBoost was relatively small, demonstrating that increased model complexity does not automatically produce a large performance gain.
11. Hyperparameter tuning was performed to determine whether XGBoost could be improved further.
12. The final model was trained on the complete training dataset before generating predictions for the unseen test data.

---

# 🏁 Conclusion

This project demonstrates a complete machine learning workflow for product-store sales prediction, from raw data preparation through final submission.

The modelling process did not rely on selecting a complex algorithm immediately. Instead, increasingly flexible models were evaluated systematically.

The results progressed from:

```text
Linear Regression
        ↓
Random Forest
        ↓
Gradient Boosting
        ↓
XGBoost
        ↓
Hyperparameter-Tuned XGBoost
```

The initial XGBoost model achieved the strongest baseline performance with:

```text
MAE  = 762.72
RMSE = 1088.40
R²   = 0.5884
```

The final tuned model was then evaluated to determine whether optimization produced a meaningful improvement.

The key outcome of the project was not simply the selection of XGBoost, but the development of a **structured, reproducible and evidence-based modelling pipeline** in which data cleaning, preprocessing, model selection, evaluation and tuning were all connected to clearly defined objectives.

---

## 📁 Suggested Repository Structure

```text
product-store-sales-prediction/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   └── sales_prediction.ipynb
│
├── outputs/
│   └── submission.csv
│
├── README.md
│
└── requirements.txt
```

---

## 📜 License

This project was developed for educational and hackathon purposes.
