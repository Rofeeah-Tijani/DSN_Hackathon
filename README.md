# DSN_Hackathon

Product-Store Sales Prediction
End-to-End Machine Learning Hackathon Documentation
1. Project Overview
This project focuses on predicting the total sales of a product at a particular store using information about the product, its pricing, visibility, and the characteristics of the store where it is sold.
The objective of the hackathon was to:
	Understand the product-store sales dataset. 
	Identify factors associated with product sales. 
	Clean and preprocess the available data. 
	Perform exploratory data analysis (EDA). 
	Engineer useful features where appropriate. 
	Build and compare machine learning regression models. 
	Evaluate the models using appropriate regression metrics. 
	Tune the strongest-performing model. 
	Use the final model to predict sales for the unseen test dataset. 
	Generate a submission file containing the required predictions. 
The target variable was:
total_sales
Because total_sales is a continuous numerical variable, this was treated as a supervised regression problem.
________________________________________
2. Understanding the Dataset
The dataset contains information at the product-store level.
Each observation represents a product being sold in a particular store, together with characteristics that may be associated with its sales.
The major variables were:
Variable	Description	Type
id	Unique row identifier	Identifier
product_code	Product identifier	Identifier
product_weight_kg	Weight of the product in kilograms	Numerical
fat_content	Product fat category	Categorical
shelf_visibility	Proportion of store display area allocated to the product	Numerical
product_category	Product category	Categorical
product_price	Listed product price	Numerical
store_code	Store identifier	Identifier
store_age_years	Number of years the store has operated	Numerical
store_size	Store size	Categorical
store_location_tier	Location classification of the store	Categorical
store_format	Format/type of store	Categorical
total_sales	Total sales of the product at the store	Target
The dataset was supplied as separate training and test datasets.
The training dataset contained total_sales, while the test dataset did not.
Therefore:
TRAIN DATA
Features + total_sales
        ↓
Learning + evaluation
        ↓
Final model
        ↓
TEST DATA
Features only
        ↓
Predicted total_sales
The test dataset was kept separate from model development and was only used for final prediction.
________________________________________
3. Initial Data Understanding
Before modelling, the dataset was examined to understand:
	the number of observations and variables; 
	data types; 
	missing values; 
	categorical values; 
	numerical distributions; 
	possible inconsistencies; 
	identifier columns; 
	the distribution of the target variable. 
This step was important because machine learning models cannot simply be applied to raw data without understanding its structure.
The initial inspection helped identify several data-quality issues, particularly:
	missing product weights; 
	missing store sizes; 
	inconsistent product-category formatting; 
	identifier columns that were not appropriate as ordinary predictive variables. 
________________________________________
4. Data Cleaning
Data cleaning was performed before modelling because the quality and consistency of the input data directly affect the reliability of the resulting model.
4.1 Handling inconsistent product-category values
One of the categorical variables contained inconsistent formatting in the product-category values.
For example, the same category could appear in different forms because of differences in capitalization or formatting.
This is a data-quality problem because a machine learning algorithm can interpret differently written versions of what should be the same category as different categories.
For example:
Food
food
FOOD
could incorrectly be interpreted as three separate categories.
Solution
A standardization function was applied to the category values so that equivalent category names were represented consistently.
The resulting standardized column was referred to as:
Proper Product_Category
The purpose was not to change the meaning of the categories, but to ensure that the same category was represented consistently throughout the dataset.
Conceptually:
Original category
        ↓
Standardization function
        ↓
Consistent category representation
        ↓
Machine learning
This was particularly important because the categorical variables were later encoded using OneHotEncoder.
Without standardization, the encoder could create separate features for categories that were actually the same.
________________________________________
5. Handling Missing Store Size
The store_size variable contained missing observations.
The possible values represented different store sizes, such as:
Small
Medium
Large
Instead of assigning a size arbitrarily, missing values were replaced with:
Unknown
Why?
Assigning Small, Medium, or Large without evidence would introduce information that was not actually present in the original data.
For example, assuming that every missing store size is Medium would artificially create a pattern that might not exist.
Using:
Unknown
preserves the fact that the information was unavailable.
It also allows the model to learn whether observations with an unknown store size behave differently from observations with known store sizes.
________________________________________
6. Handling Missing Product Weight
product_weight_kg also contained missing values.
A simple overall median could have been used, but the approach adopted was group-wise median imputation.
Instead of replacing every missing weight with one global median, the missing value was replaced using the median weight of the relevant group.
Conceptually:
Product group
      ↓
Find available weights within that group
      ↓
Calculate group median
      ↓
Use that median for missing weight
Why group-wise median?
Products belonging to different groups can naturally have different typical weights.
For example, the typical weight of one product category may be very different from another.
Using a single global median could therefore ignore meaningful differences between groups.
Group-wise median imputation preserves more of the structure of the dataset.
Why median rather than mean?
The median is less sensitive to extreme values.
If a group contains:
1 kg
1.2 kg
1.3 kg
1.5 kg
20 kg
the mean would be strongly affected by the 20 kg observation, while the median would provide a more representative central value.
Therefore, group-wise median imputation was considered more appropriate for the missing product-weight observations.
________________________________________
7. Identifier Columns
Three columns were excluded from the modelling features:
id
product_code
store_code
The modelling dataset was therefore created as:
X = train.drop(
    columns=['total_sales', 'id', 'product_code', 'store_code']
)

y = train['total_sales']
Why was id removed?
id is simply an identifier for the observation.
It does not represent a meaningful product or store characteristic.
Including it could allow the model to learn arbitrary numerical patterns from the identifier rather than meaningful relationships with sales.
________________________________________
Why were product_code and store_code removed?
They were treated as identifiers rather than direct measurements of product or store characteristics.
The objective was to build the model using the descriptive variables supplied in the dataset, such as:
	product weight; 
	price; 
	visibility; 
	category; 
	store size; 
	location tier; 
	store format; 
	store age. 
This also reduced the risk of the model treating arbitrary identifier values as numerical information.
________________________________________
8. Final Modelling Features
After cleaning and removing identifiers, the modelling features were:
Numerical features
product_weight_kg
shelf_visibility
product_price
store_age_years
Categorical features
fat_content
Proper Product_Category
store_size
store_location_tier
store_format
Target:
total_sales
________________________________________
9. Exploratory Data Analysis
EDA was performed before modelling to understand the structure of the data and investigate relationships that could potentially be useful for prediction.
The analysis was divided into several areas.
________________________________________
9.1 Target Variable Analysis
The distribution of:
total_sales
was examined using descriptive statistics and visualizations.
This helped determine:
	central tendency; 
	spread; 
	possible skewness; 
	presence of extreme observations; 
	overall sales distribution. 
This was important because the target distribution influences how prediction errors should be interpreted.
________________________________________
10. Numerical Variables vs Sales
The numerical variables were examined against total_sales.
These included:
product_weight_kg
shelf_visibility
product_price
store_age_years
Scatterplots and correlation analysis were used to investigate whether changes in these variables were associated with changes in sales.
The purpose was not to assume that correlation meant causation.
Instead, the EDA asked:
"Is there a noticeable relationship or pattern between this variable and sales that the models may be able to learn?"
________________________________________
11. Categorical Variables vs Sales
Categorical variables were also investigated.
These included:
fat_content
Proper Product_Category
store_size
store_location_tier
store_format
Group-level sales statistics and visualizations were used to determine whether sales distributions differed across categories.
For example:
Store Format
      ↓
Average sales by format
and:
Location Tier
      ↓
Average sales by location tier
This helped us understand how sales differed across different product and store characteristics.
________________________________________
12. Interaction Analysis
Relationships between combinations of variables were also explored.
Examples included:
	store format × location tier; 
	product category × store format; 
	product category × location tier; 
	price × shelf visibility. 
This was important because sales may not depend on one variable in isolation.
For example, the relationship between product price and sales could potentially differ depending on where or in what type of store the product is sold.
The purpose of these analyses was to understand the data more deeply and identify patterns that a nonlinear model might be able to capture.
________________________________________
13. Feature Engineering
Feature engineering was considered because the hackathon specifically encouraged the creation of useful features.
However, feature engineering should not be performed simply for the sake of creating more columns.
After examining the available variables and their meanings, no additional engineered feature was considered sufficiently justified to introduce into the final modelling pipeline.
Therefore, the modelling process proceeded using the cleaned original features.
This was a deliberate decision:
Not creating an artificial feature is better than creating a feature without a defensible reason.
________________________________________
14. Data Preprocessing
Machine learning algorithms cannot directly process categorical text variables such as:
Small
Medium
Large
or:
Superstore
Corner Shop
in their raw form.
Therefore, preprocessing was required.
________________________________________
14.1 One-Hot Encoding
Categorical variables were converted using:
OneHotEncoder(handle_unknown='ignore')
For example:
store_size
could be transformed into separate binary features such as:
store_size_Small
store_size_Medium
store_size_Large
store_size_Unknown
Similarly, store_format was represented using separate encoded features.
Why One-Hot Encoding?
The categories do not necessarily represent numerical quantities.
For example:
Corner Shop = 1
Supermarket = 2
Superstore = 3
would incorrectly suggest that Superstore is mathematically "greater" than Corner Shop.
One-hot encoding avoids imposing such an artificial numerical relationship.
________________________________________
15. Why handle_unknown='ignore'?
This was especially important because the final model would eventually be applied to the test data.
The test set could potentially contain a categorical value that was not encountered while fitting the encoder on the training data.
With:
OneHotEncoder(handle_unknown='ignore')
the pipeline can handle such unseen categories without failing.
This makes the preprocessing pipeline safer for the final prediction stage.
________________________________________
16. Numerical Preprocessing
For the tree-based models, numerical variables were passed through without standard scaling:
('num', 'passthrough', numerical_features)
This was because tree-based models such as:
	Random Forest; 
	Gradient Boosting; 
	XGBoost 
do not require numerical features to be standardized in the same way that many linear or distance-based models do.
For the Linear Regression baseline, standard scaling was applied to numerical variables because scaling provides a more consistent numerical representation for the linear modelling pipeline.
________________________________________
17. Why Use a Pipeline?
The preprocessing and model were combined into a single:
Pipeline
This was important because it ensured that preprocessing was performed consistently whenever the model was trained or used for prediction.
Conceptually:
Raw data
   ↓
Preprocessing
   ↓
Encoded features
   ↓
Model
   ↓
Prediction
The pipeline also helps prevent data leakage during cross-validation because preprocessing is fitted within each training fold rather than being fitted once using information from the entire dataset.
________________________________________
18. Model Development Strategy
Four regression algorithms were evaluated:
	Linear Regression 
	Random Forest Regressor 
	Gradient Boosting Regressor 
	XGBoost Regressor 
The models were evaluated using 5-fold cross-validation.
________________________________________
19. Why Linear Regression?
Linear Regression was used as the baseline model.
The idea was simple:
Before using more sophisticated algorithms, establish how well a relatively simple model can predict sales.
Linear Regression assumes that the relationship between the predictors and the target can be reasonably represented through a linear combination of the features.
Its main advantages include:
	simplicity; 
	interpretability; 
	fast training; 
	useful baseline performance. 
However, product-store sales relationships may not be purely linear.
Therefore, more flexible models were also investigated.
________________________________________
20. Linear Regression Results
The 5-fold cross-validation results were:
Metric	Result
MAE	839.80
RMSE	1129.73
R²	0.5565
Interpretation
The MAE of approximately 839.80 means that, on average, the model's absolute prediction error was around 840 sales units.
The RMSE of approximately 1129.73 was higher than the MAE, indicating that some predictions had substantially larger errors than the typical prediction.
The R² of approximately 0.5565 means that the model explained about 55.65% of the variation in the target under cross-validation.
Importantly, R² is not accuracy.
This provided our baseline against which the more flexible models could be compared.
________________________________________
21. Why Random Forest?
The next model was Random Forest Regression.
Random Forest builds many decision trees and combines their predictions.
Unlike Linear Regression, it can capture:
	nonlinear relationships; 
	interactions between variables; 
	more complex patterns. 
This made it a logical next step because sales relationships may not be linear.
The model used:
RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
________________________________________
22. Random Forest Results
Metric	Linear Regression	Random Forest
MAE	839.80	781.33
RMSE	1129.73	1119.93
R²	0.5565	0.5642
Random Forest improved on Linear Regression across all three metrics.
MAE
The MAE decreased from:
839.80 → 781.33
This means the typical absolute prediction error was reduced.
RMSE
RMSE decreased from:
1129.73 → 1119.93
indicating a modest reduction in larger prediction errors.
R²
R² increased from:
0.5565 → 0.5642
This indicates that Random Forest captured slightly more of the variation in sales.
The improvement demonstrated that nonlinear tree-based modelling was useful for the problem.
________________________________________
23. Why Gradient Boosting?
The next model was Gradient Boosting Regression.
Gradient Boosting works differently from Random Forest.
Rather than building many independent trees and averaging them, Gradient Boosting builds trees sequentially.
Each subsequent tree attempts to improve upon the errors made by the previous trees.
This makes Gradient Boosting particularly useful when there are complex nonlinear relationships that simpler models may not capture.
The model used:
GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
________________________________________
24. Gradient Boosting Results
Metric	Random Forest	Gradient Boosting
MAE	781.33	765.43
RMSE	1119.93	1089.83
R²	0.5642	0.5873
Gradient Boosting produced a clearer improvement.
The MAE decreased from:
781.33 → 765.43
RMSE decreased from:
1119.93 → 1089.83
while R² increased from:
0.5642 → 0.5873
This showed that sequential boosting was capturing additional patterns that Random Forest was not capturing as effectively under the tested configurations.
________________________________________
25. Why XGBoost?
XGBoost was then introduced as a more optimized and powerful gradient-boosting implementation.
XGBoost is designed to provide:
	strong predictive performance; 
	regularization; 
	efficient tree construction; 
	control over model complexity; 
	handling of nonlinear relationships and feature interactions. 
Because Gradient Boosting had already shown strong performance, XGBoost was a logical next model to investigate rather than arbitrarily testing algorithms.
The initial XGBoost configuration was:
XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
    objective='reg:squarederror',
    n_jobs=-1
)
________________________________________
26. XGBoost Results
The initial XGBoost model achieved:
Model	MAE ↓	RMSE ↓	R² ↑
Linear Regression	839.80	1129.73	0.5565
Random Forest	781.33	1119.93	0.5642
Gradient Boosting	765.43	1089.83	0.5873
XGBoost	762.72	1088.40	0.5884
XGBoost produced the strongest baseline performance across all three metrics.
However, an important observation is that the improvement over Gradient Boosting was small.
XGBoost vs Gradient Boosting
MAE:
765.43 → 762.72
Improvement ≈ 2.71
RMSE:
1089.83 → 1088.40
Improvement ≈ 1.44
R²:
0.5873 → 0.5884
Improvement ≈ 0.0011
Therefore, it would be inaccurate to claim that XGBoost dramatically outperformed Gradient Boosting.
Instead:
XGBoost produced the best baseline cross-validation performance, but only by a narrow margin over Gradient Boosting.
That distinction is important in proper hackathon documentation.
________________________________________
27. Why Hyperparameter Tuning?
The initial XGBoost model used manually selected hyperparameters:
n_estimators = 300
learning_rate = 0.05
max_depth = 3
There was no guarantee that these were the most suitable settings for this dataset.
Therefore, hyperparameter tuning was performed.
The purpose was:
Find a better combination of XGBoost settings for this particular dataset rather than assuming that the initial settings were optimal.
________________________________________
28. RandomizedSearchCV
RandomizedSearchCV was used instead of manually testing parameter combinations.
The search space included:
n_estimators
learning_rate
max_depth
min_child_weight
subsample
colsample_bytree
reg_alpha
reg_lambda
These parameters control different aspects of XGBoost, including:
	number of trees; 
	learning speed; 
	tree complexity; 
	minimum amount of information required for further splitting; 
	proportion of observations used; 
	proportion of features used; 
	regularization. 
Thirty parameter combinations were sampled.
Each combination was evaluated using 5-fold cross-validation.
Therefore:
30 combinations × 5 folds
= 150 model fits
________________________________________
29. Why MAE Was Used During Tuning
The tuning objective was:
scoring='neg_mean_absolute_error'
This means the search was trying to identify the XGBoost configuration with the best MAE.
MAE was selected because it is easy to interpret:
On average, how far are the predictions from the actual sales values?
Because scikit-learn treats loss metrics as quantities to maximize during its internal search, MAE appears as a negative value internally.
Therefore:
best_cv_mae = -xgb_tuning.best_score_
converts it back to the normal positive MAE value.
________________________________________
30. Why We Still Evaluated RMSE and R²
Although MAE was used to guide the hyperparameter search, the final evaluation did not rely on MAE alone.
The tuned model was evaluated using:
	MAE; 
	RMSE; 
	R². 
This provides a more complete picture.
MAE
Measures the average absolute error.
Lower is better.
RMSE
Measures prediction error while placing greater emphasis on larger errors.
Lower is better.
R²
Measures how much of the variation in the target is explained by the model relative to a baseline.
Higher is better.
________________________________________
31. Model Evaluation Summary
The complete baseline comparison was:
Model	MAE ↓	RMSE ↓	R² ↑
Linear Regression	839.80	1129.73	0.5565
Random Forest	781.33	1119.93	0.5642
Gradient Boosting	765.43	1089.83	0.5873
XGBoost	762.72	1088.40	0.5884
The progression demonstrates an important pattern:
Linear Regression
       ↓
Random Forest
       ↓
Gradient Boosting
       ↓
XGBoost
As the models became more capable of representing nonlinear relationships and interactions, predictive performance improved.
The largest improvement came from moving away from the linear baseline toward tree-based boosting models.
The difference between Gradient Boosting and XGBoost, however, was relatively small.
________________________________________
32. Model Selection
The initial XGBoost model was selected for further optimization because it achieved the strongest cross-validation performance across:
	MAE; 
	RMSE; 
	R². 
The selection was therefore based on the observed validation results rather than simply choosing XGBoost because it is a popular algorithm.
At the same time, Gradient Boosting remained a strong alternative because its performance was very close to XGBoost.
________________________________________
33. Feature Importance
After obtaining the tuned XGBoost model, feature importance was examined.
The purpose was to understand which processed features contributed most strongly to the model's predictions.
Because categorical variables were one-hot encoded, a categorical variable such as:
store_format
could become several separate model features.
Therefore, feature importance was extracted after preprocessing.
This allowed us to examine what the actual trained model was using.
However, feature importance should be interpreted carefully.
A high feature importance means:
The feature was useful to the model when making predictions.
It does not automatically mean:
The feature causes higher sales.
Therefore, the results should be described in terms of model importance or predictive association, rather than causation.
________________________________________
34. Final Training Strategy
Once model selection and tuning were completed, the final selected pipeline was fitted using the complete training dataset:
All available training observations
              ↓
      Final preprocessing
              ↓
       Tuned XGBoost
              ↓
       Final trained model
This was appropriate because the model had already been evaluated during development.
Using all training observations allows the final model to learn from as much labelled information as possible before generating predictions for the unseen test dataset.
________________________________________
35. Test Data Preparation
The test data was processed using the same preprocessing structure used for training.
The same modelling columns were retained, while:
id
product_code
store_code
were excluded from the predictive feature matrix.
The test feature names were also checked against the training feature names.
This was particularly important because a column naming inconsistency was discovered:
Training:
Proper Product_Category
while the test dataset initially contained:
Proper Product Category 
The difference consisted of both an underscore/space difference and a trailing space.
The test column was therefore renamed to exactly match the training feature:
X_test = X_test.rename(
    columns={
        'Proper Product Category ': 'Proper Product_Category'
    }
)
The columns were then verified using:
X.columns.equals(X_test.columns)
The result needed to be:
True
before predictions could safely be generated.
This check is important because machine-learning pipelines expect the same feature structure during prediction as was present during training.
________________________________________
36. Final Prediction
After fitting the final model:
best_xgb_model.fit(X, y)
predictions were generated:
test_predictions = best_xgb_model.predict(X_test)
The model automatically performed the necessary preprocessing because preprocessing and XGBoost were contained within the same pipeline.
The output was a predicted total_sales value for each test observation.
________________________________________
37. Submission File
The test IDs were preserved separately:
test_ids = test['id']
The final submission was created as:
submission = pd.DataFrame({
    'id': test_ids,
    'total_sales': test_predictions
})
and saved as:
submission.to_csv(
    'submission.csv',
    index=False
)
The resulting file contained:
id | total_sales
and was prepared for submission to the hackathon evaluation platform.
________________________________________
38. Complete Project Architecture
The entire project can be represented as the following architecture:
                         PRODUCT-STORE DATA
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
             TRAIN DATA                   TEST DATA
                 │                             │
        Contains total_sales              No target
                 │                             │
                 ▼                             │
          DATA UNDERSTANDING                    │
                 │                             │
                 ▼                             │
          DATA QUALITY CHECK                    │
                 │                             │
       ┌─────────┼──────────┐                  │
       │         │          │                  │
       ▼         ▼          ▼                  │
 Missing     Inconsistent  Identifier          │
 Values      Categories    Columns             │
       │         │          │                  │
       ▼         ▼          ▼                  │
 Group-wise  Category      Remove             │
 Median      Standardize   IDs                │
 Imputation                                       │
       │         │          │                  │
       └─────────┼──────────┘                  │
                 ▼                             │
            CLEAN DATA                          │
                 │                             │
                 ▼                             │
                 EDA                           │
                 │                             │
       ┌─────────┼────────────┐                │
       │         │            │                │
       ▼         ▼            ▼                │
     Target   Numerical    Categorical          │
   Analysis   Analysis      Analysis            │
       │         │            │                │
       └─────────┼────────────┘                │
                 ▼                             │
         INTERACTION ANALYSIS                 │
                 │                             │
                 ▼                             │
        FEATURE ENGINEERING                    │
                 │                             │
       No additional feature                  │
       considered sufficiently                │
       justified                              │
                 │                             │
                 ▼                             │
          PREPROCESSING                        │
                 │                             │
        ┌────────┴────────┐                    │
        │                 │                    │
   Numerical          Categorical              │
   Features           Features                 │
        │                 │                    │
        │          One-Hot Encoding            │
        │                 │                    │
        └────────┬────────┘                    │
                 ▼                             │
              PIPELINE                         │
                 │                             │
                 ▼                             │
       ┌─────────────────────┐                │
       │  MODEL COMPARISON    │                │
       └─────────────────────┘                │
                 │                             │
       ┌─────────┼──────────┬──────────┐       │
       ▼         ▼          ▼          ▼       │
     Linear   Random     Gradient    XGBoost   │
   Regression Forest     Boosting               │
       │         │          │          │       │
       └─────────┴──────────┴──────────┘       │
                         │                     │
                         ▼                     │
                  5-FOLD CV                    │
                         │                     │
                         ▼                     │
                  METRIC COMPARISON             │
                         │                     │
                         ▼                     │
                    XGBoost                    │
                  Best Baseline                │
                         │                     │
                         ▼                     │
              HYPERPARAMETER TUNING            │
                         │                     │
                 RandomizedSearchCV             │
                         │                     │
              30 combinations × 5 folds        │
                         │                     │
                         ▼                     │
                 TUNED XGBOOST                 │
                         │                     │
                         ▼                     │
             MAE + RMSE + R² Evaluation        │
                         │                     │
                         ▼                     │
                 FEATURE IMPORTANCE            │
                         │                     │
                         └──────────────┐      │
                                        │      │
                                        ▼      ▼
                              FINAL MODEL
                                        │
                                        ▼
                              PREPARE TEST DATA
                                        │
                                        ▼
                              SAME PREPROCESSING
                                        │
                                        ▼
                                TEST PREDICTION
                                        │
                                        ▼
                                total_sales
                                        │
                                        ▼
                              submission.csv
________________________________________
39. Why This Workflow Is Appropriate for the Hackathon
The approach followed a progressive modelling strategy rather than immediately jumping to a complex algorithm.
Stage 1 — Understand
We first established what each variable represented and identified problems in the raw data.
Stage 2 — Clean
Missing and inconsistent information was handled using approaches appropriate to the nature of each variable.
Stage 3 — Explore
EDA was used to understand sales distributions, relationships, category differences, and interactions.
Stage 4 — Prepare
The data was transformed into a format suitable for machine learning while maintaining a consistent preprocessing pipeline.
Stage 5 — Establish a baseline
Linear Regression provided a reference point.
Stage 6 — Increase model complexity
Random Forest was introduced to capture nonlinear relationships.
Stage 7 — Boost
Gradient Boosting was introduced to sequentially improve prediction errors.
Stage 8 — Optimize
XGBoost was tested as a more advanced gradient-boosting implementation.
Stage 9 — Tune
RandomizedSearchCV searched for better XGBoost hyperparameters rather than relying solely on manually chosen values.
Stage 10 — Interpret
Feature importance was examined to understand which processed variables the model relied upon most.
Stage 11 — Finalize
The final model was trained on all available training data and used to predict the unseen test data.
Stage 12 — Submit
Predictions were combined with the required IDs and exported into the submission CSV.
________________________________________
40. Key Lessons From the Model Comparison
One of the most important findings from the modelling process was that more complex models did not automatically produce dramatically better results.
The progression was:
Linear Regression
MAE  = 839.80
RMSE = 1129.73
R²   = 0.5565

        ↓

Random Forest
MAE  = 781.33
RMSE = 1119.93
R²   = 0.5642

        ↓

Gradient Boosting
MAE  = 765.43
RMSE = 1089.83
R²   = 0.5873

        ↓

XGBoost
MAE  = 762.72
RMSE = 1088.40
R²   = 0.5884
The biggest improvement occurred when moving from the linear baseline to nonlinear tree-based models.
The improvement from Gradient Boosting to XGBoost was relatively small.
This suggests that the dataset contains nonlinear relationships that tree-based models can capture, but the additional sophistication of XGBoost does not produce an enormous improvement over Gradient Boosting under the tested configurations.
That is a valuable technical finding in itself.
________________________________________
41. Metric Interpretation for the Final Report
The three metrics should be clearly explained in the hackathon documentation.
Mean Absolute Error — MAE
MAE=1/n∑∣y_i-(y_i ) ̂∣

MAE represents the average absolute difference between actual and predicted sales.
For the initial XGBoost model:
MAE ≈ 762.72
Therefore, the model's predictions were off by approximately 763 sales units on average in absolute terms under the 5-fold cross-validation evaluation.
Lower MAE is better.
________________________________________
Root Mean Squared Error — RMSE
RMSE=√(1/n∑(y_i-(y_i ) ̂)^2 )

RMSE gives greater weight to larger errors because the errors are squared before averaging.
For XGBoost:
RMSE ≈ 1088.40
The fact that RMSE is considerably higher than MAE suggests that some observations have relatively large prediction errors.
Lower RMSE is better.
________________________________________
R² — Coefficient of Determination
R^2=1-(∑(y_i-(y_i ) ̂)^2)/(∑(y_i-y ˉ)^2 )

The initial XGBoost model achieved:
R² ≈ 0.5884
This means that approximately 58.84% of the variation in total_sales was explained by the model under the cross-validation evaluation.
It does not mean:
58.84% prediction accuracy
and it does not mean that the remaining 41.16% is necessarily "error."
It is a measure of explained variance relative to a baseline model that predicts the mean.
Higher R² is better.
________________________________________
42. Final Technical Conclusion
The project developed an end-to-end machine learning pipeline for predicting product-store sales.
The process began with data understanding and quality assessment, followed by cleaning of inconsistent product-category values, handling of missing store sizes using Unknown, and group-wise median imputation for missing product weights.
Identifier variables were excluded from the modelling features, while meaningful product and store characteristics were retained.
EDA was then used to investigate the distribution of sales, numerical relationships, categorical differences, and potential interactions.
The cleaned data was transformed using a preprocessing pipeline, with categorical variables one-hot encoded and numerical variables handled appropriately for each model family.
Four regression approaches were evaluated using 5-fold cross-validation:
	Linear Regression; 
	Random Forest; 
	Gradient Boosting; 
	XGBoost. 
Linear Regression established the baseline with an MAE of 839.80, RMSE of 1129.73, and R² of 0.5565.
Random Forest improved these results to 781.33 MAE, 1119.93 RMSE, and 0.5642 R².
Gradient Boosting produced a further improvement, achieving 765.43 MAE, 1089.83 RMSE, and 0.5873 R².
XGBoost achieved the strongest baseline performance, with 762.72 MAE, 1088.40 RMSE, and 0.5884 R².
Because XGBoost produced the best baseline results across all three evaluation metrics, it was selected for hyperparameter optimization. RandomizedSearchCV was then used to investigate different combinations of XGBoost hyperparameters through 5-fold cross-validation.
The tuned model was subsequently evaluated using MAE, RMSE, and R², and feature importance was examined to understand the variables contributing most strongly to the model's predictions.
Finally, after model development and tuning were completed, the selected pipeline was fitted using the full training dataset. The same preprocessing structure was applied to the unseen test data, predictions for total_sales were generated, and the results were combined with the test IDs to create the final submission.csv.
The overall architecture therefore followed:
Data Understanding → Data Cleaning → EDA → Feature Assessment → Preprocessing → Baseline Modelling → Cross-Validation → Model Comparison → XGBoost Selection → Hyperparameter Tuning → Model Interpretation → Final Training → Test Prediction → Submission
Hackathon
