## 📁 Repository Structure
- `/steel_production_analysis/`
    - `data/`
        - `normalized_test_data.csv`
        - `normalized_train_data.csv`
    - `figures/`
        - `eda/`
          - `box_plots.png`
          - `correlation_matrix.png`
          - `feature_distributions.png`
          - `pair_plot.png`
          - `target_distributions.png`
        - `model_performances/`
          - `gpr_predictions_vs_actual.png`
          - `gpr_residuals.png`
          - `mlp_predictions_vs_actual.png`
          - `mlp_residuals.png`
          - `Model Performance Comparison.png`
          - `randomforest_predictions_vs_actual.png`
          - `randomforest_residuals.png`
          - `svr_predictions_vs_actual.png`
          - `svr_residuals.png`
        - `training_performances/`
          - `gpr_learning_curve.png`
          - `mlp_learning_curve.png`
          - `randomforest_learning_curve.png`
          - `svr_learning_curve.png`
    - `results/`
        - `log/`
          - `Training_log.txt`
        - `models/`
          - `gpr_final.joblib`
          - `mlp_final.joblib`
          - `randomforest_final.joblib`
          - `svr_final.joblib`
        - `table/`
          - `test_performances.csv`
    - `scripts/`
        - `data_loading.py`
        - `data_preprocessing.py`
        - `eda.py`
        - `model_training.py`
        - `results_analysis.py`
    - `.gitattributes`
    - `.gitignore`
    - `README.md`

**Explanation of the structure:**

- `data/`: Contains all the data used in the project.
- `figures/`: Contains all visual outputs generated during the project
- `results/`: Contains the outputs of your analyses.
- `scripts/`: Contains Python source code for the complete machine learning pipeline.
- `.gitattributes`:Configures Git LFS to manage large binary files in this repository.
- `.gitignore`: Lists files/directories to ignore.
- `README.md`: This file, containing information about the project.

## Project 1

### Steel Production Analysis

### Abstract

This project explores the application of machine learning models to predict outcomes in a steel production process. A normalized dataset is analyzed using several regression methods, including Random Forest, Support Vector Regression, Gaussian Process Regression, and Multilayer Perceptron. The main objective is to evaluate and compare the predictive performance of these models. Through this process, the project demonstrates how data preprocessing, model training, and evaluation are applied in an industrial context.

### Introduction

**Background**

The steel production dataset consists of 21 input features describing process variables and a single target variable representing final steel quality. The aim is to apply machine learning methods to analyze and preprocess the data and to construct predictive models that estimate the final quality from the inputs. The workflow follows a standard course pattern: data cleaning and normalization, feature engineering, training multiple regression algorithms (e.g., Random Forest, Support Vector Regression, Gaussian Process Regression, Multilayer Perceptron), and evaluating predictive performance to support learning outcomes.

**Objectives**

- To preprocess and analyze steel production data using exploratory data analysis techniques.
- To train multiple regression models for predicting production outcomes.
- To evaluate and compare model performance using appropriate metrics.
- To gain practical experience in applying machine learning methods to an industrial dataset.

## Data Description

#### Data Overview

- **Data composition**: 21 numeric input features (input1–input21) and 1 target (output).
- **Data sources and formats**: Two normalized CSV files were used — normalized_train_data.csv (7642 rows, 22 columns) and normalized_test_data.csv (3337 rows, 22 columns). After concatenation, the combined dataset contains 10,979 samples and 22 columns (21 inputs + 1 output).
- **Feature distribution**: Visual inspection of the five figures indicates that input features and the target are scaled roughly to the [0, 1] range, consistent with normalization.
- **Target description**: Output denotes the final steel quality indicator and is treated as a numeric regression target.

#### Data Quality and Assumptions

- **Missing values**: No missing values detected during preprocessing.

- **Duplicates**: No duplicate rows detected.

- **Outliers**: Multiple features exhibit outliers. Outliers are mitigated using the interquartile range (IQR) method with each column’s median as the replacement value. Example counts from preprocessing logs include input2 (537), input5 (906), input10 (698), etc.

- **Assumptions**: The data reflect actual production conditions; normalization supports stable model training and generalization.

#### EDA

- **Figure 1: Box plot of features**
  ![box_plots](C:\Users\admin\OneDrive\Desktop\steel_production_analysis\figures\eda\box_plots.png)
  - **Observations**: Outliers are present across many features; the range and dispersion vary by feature. This supports the need for robust preprocessing and potentially robust scaling.
  
- **Figure 2: Correlation Matrix**
  ![correlation_matrix](C:\Users\admin\OneDrive\Desktop\steel_production_analysis\figures\eda\correlation_matrix.png)
  
  - **Observations**: Inter-feature correlations are mixed. Some pairs show moderate positive correlations, others are weakly related. The output (target) shows varying associations with inputs, indicating that several features may contribute to the target in a non-uniform manner.
  
- **Figure 3: Feature distributions (histogram grid)**
  ![feature_distributions](C:\Users\admin\OneDrive\Desktop\steel_production_analysis\figures\eda\feature_distributions.png)
  
  - **Observations**: Most features cluster within [0, 1], with distributions ranging from near-Gaussian to mildly skewed shapes. The variation across features suggests different underlying data-generating processes and warrants nonlinear modeling considerations for some features.
  
- **Figure 4: Pair Plot**
  ![pair_plot](C:\Users\admin\OneDrive\Desktop\steel_production_analysis\figures\eda\pair_plot.png)
  
  - **Observations**: Nonlinear relationships and interactions are evident between several input pairs. The relation between inputs and the target appears multi-factorial and not strictly linear.
  
- **Figure 5: Target distributions**
  ![target_distributions](C:\Users\admin\OneDrive\Desktop\steel_production_analysis\figures\eda\target_distributions.png)
  
  - **Observations**: The output distribution is unimodal and centered around the mid-range of the normalized scale, with a slight tail toward higher values. This supports a standard regression approach, though minor skewness/kurtosis could influence error distribution.

#### Reproducibility and Environment

- **Data files**: normalized_train_data.csv, normalized_test_data.csv
- **Preprocessing script**: data_preprocessing.py (defines RemoveDuplicates, HandleMissingValues, DetectOutliers, EncodeCategoricalVariables, StandardScaler)
- **Model training script**: model_training.py (handles data splits and downstream modeling steps)
- **Libraries and environment**: Python, pandas, NumPy, scikit-learn (explicit version information should be recorded in the project environment)
- **Random seed**: 42 (ensures reproducible train/validation/test splits)
- **Logging and outputs**: Preprocessing logs record duplicates, missing values (if any), outlier replacements, and per-step data shapes to facilitate reproducibility.

## Methods

This section describes the data processing, model training, and evaluation strategy for the applied machine learning project. It covers preprocessing, feature engineering attempts, hyperparameter optimization, learning-curve analysis, and the final evaluation plan.

#### Data preprocessing

- **To avoid data leakage and to manage preprocessing consistently across splits, all preprocessing steps are implemented as scikit-learn Transformers and composed into a single Pipeline.** This ensures that transformations are fitted only on the training data and then applied to validation and test data.
- **The preprocessing pipeline (data_preprocessing_pipeline) includes:**
  - RemoveDuplicates: drops duplicate rows.
  - HandleMissingValues: imputes numeric features with column medians.
  - DetectOutliers: uses the IQR method to identify outliers and replaces them with the column median.
  - EncodeCategoricalVariables: applies one-hot encoding to categorical features.
  - StandardScaler: standardizes features after encoding.
- **Data splitting follows a typical train/validation/test scheme (e.g., 0.7/0.15/0.15).** The target column is named `output`. The pipeline is fitted on the training subset and applied to validation and test subsets, ensuring no information from test data leaks into model training.

- **Detect outliers by IQR method:** For numeric features, we compute Q1 and Q3, derive IQR, and set lower/upper bounds as:
  - lower = Q1 − 1.5 × IQR
  - upper = Q3 + 1.5 × IQR
- **Outliers (values outside [lower, upper]) are replaced by the feature’s median.** This approach is simple, robust, and keeps the data distribution stable for subsequent modeling.

#### Model training and hyperparameter optimization

- **Models considered:**
  - RandomForestRegressor
  - SVR (Support Vector Regression)
  - MLPRegressor (Neural network)
  - GaussianProcessRegressor (GPR)
- **Hyperparameter optimization methods:**
  - GridSearchCV: exhaustive search over a specified parameter grid.
  - RandomizedSearchCV: stochastic search over parameter distributions.
- **Cross-validation:**
  - 5-fold CV is used for hyperparameter tuning.
  - Scoring for CV is neg_mean_squared_error (lower is better; converted to RMSE/R2 for reporting as needed).
- **Parameter spaces (highlights):**
  - RandomForestRegressor: n_estimators, max_features, max_depth, min_samples_leaf, min_samples_split, bootstrap
  - SVR: C, epsilon, kernel, degree, gamma
  - MLPRegressor: learning_rate, hidden_layer_sizes, activation, batch_size, alpha, solver
  - GaussianProcessRegressor: kernel (e.g., RBF, Matérn combinations), alpha, normalize_y, n_restarts_optimizer
- **Training Logs and Experiment Traceability**: 
  - Objective: Record the best hyperparameters and the corresponding validation performance for each training run to ensure reproducibility.
  - Log location: The code appends to base_dir/results/log/Training_log.txt (append mode).
  - Logged fields: Time, Model, Search Method, CV, Scoring, Best Params, Best CV Score, Train Metrics, and Validation Metrics.
  - Purpose: Create an auditable trail to compare configurations across runs.

- **After identifying the best hyperparameters, the best model is refitted on the full training data subset (training + validation as appropriate) before final evaluation on the test set.**

#### Feature engineering experiments and outcomes

- **RFE (Recursive Feature Elimination):** attempted to reduce feature dimensionality to improve generalization. Result: no meaningful improvement in validation performance.
- **PCA with Gaussian Process Regressor (GPR):** attempted to reduce dimensionality to mitigate overfitting. Result: led to substantial overfitting and degraded validation performance.
- **PolynomialFeatures:** introduced polynomial feature expansion to capture nonlinear patterns. Result: did not improve validation performance.
- **Overall conclusion:** simple feature scaling and one-hot encoding plus standard models were insufficient to overcome the observed generalization gap with the current data; more sophisticated feature engineering or data collection may be required.

#### Learning-curve analysis (four learning curves)

- **Gaussian Process Regressor Data-Size Learning Curve**
  ![gpr_learning_curve](C:\Users\admin\OneDrive\Desktop\steel_production_analysis\figures\training_performances\gpr_learning_curve.png)
  - Training RMSE decreases as the training fraction increases, showing the model benefits from more data.
  - Validation RMSE remains relatively high (roughly 0.072–0.078 across fractions), indicating limited generalization with this setup.
- **Random Forest Regressor Data-Size Learning Curve**
  ![randomforest_learning_curve](C:\Users\admin\OneDrive\Desktop\steel_production_analysis\figures\training_performances\randomforest_learning_curve.png)
  - Training RMSE improves with more data; validation RMSE gradually decreases from ~0.075 toward ~0.065 as data size grows.
  - The gap between training and validation narrows with more data, suggesting modest improvements in generalization.
- **SVR Data-Size Learning Curve**
  ![svr_learning_curve](C:\Users\admin\OneDrive\Desktop\steel_production_analysis\figures\training_performances\svr_learning_curve.png)
  - Training RMSE is low and improves with data; validation RMSE remains around ~0.073–0.076, similar to RF, indicating a persistent generalization gap.
- **MLP Regressor Iteration Learning Curve**
  ![mlp_learning_curve](C:\Users\admin\OneDrive\Desktop\steel_production_analysis\figures\training_performances\mlp_learning_curve.png)
  - Training loss decreases rapidly with iterations, showing fast convergence.
  - The figure indicates strong optimization on training data, but without a clear validation curve, generalization behavior remains uncertain.
- **Overall interpretation:**
  - None of the four models achieves strong generalization on the validation set.
  - The observed performance gap suggests underfitting or intrinsic data limitations; more advanced feature engineering or alternative modeling approaches may be necessary.

Note: The learning-curve figures provided in the report are used to guide interpretation of model behavior with varying data sizes and training iterations.

#### Final training and evaluation protocol

- **Model selection:**
  - After hyperparameter search, select the best-performing model according to the validation metrics (e.g., MSE-based or R2-based).
- **Re-training on the full training+validation set:**
  - Once the optimal model is identified, re-train it on the combined training and validation data.
- **Final evaluation on the test set:**
  - Evaluate the retrained model on the held-out test set.
  - Metrics (to be reported in a results table) may include R2, RMSE, MAE, and training/inference times.
- **Results presentation:**
  - The test-set results will be summarized in a tabular format (see placeholder table below).

#### Placeholder for test-set results

| Model        | RMSE   | MAE    | $R^2$  | Training Time (s) | Inference Time (s) |
| ------------ | ------ | ------ | ------ | ----------------- | ------------------ |
| RandomForest | 0.0657 | 0.0478 | 0.4695 | 4.4233            | 0.9559             |
| SVR          | 0.0735 | 0.0549 | 0.3361 | 4.3818            | 0.8031             |
| MLP          | 0.0775 | 0.0596 | 0.2611 | 27.1585           | 0.0096             |
| GPR          | 0.0732 | 0.0551 | 0.3419 | 765.4008          | 0.6508             |

Subsequent sections will fill these fields after the final evaluation on the test set.

#### Tools Used

- Language: Python 3.9
- Data handling: pandas, numpy
- Machine learning framework: scikit-learn
  - Core components: Pipeline, ColumnTransformer (if used), Transformers (for scaling and encoding)
  - Preprocessing: StandardScaler, OneHotEncoder
  - Estimators (models): RandomForestRegressor, SVR (Support Vector Regressor), MLPRegressor (Multilayer Perceptron), GaussianProcessRegressor
  - Hyperparameter tuning: GridSearchCV, RandomizedSearchCV
  - Evaluation: cross-validation, common regression metrics (e.g., RMSE, R2)
- Experiment tracking and logging:
  - Logging to base_dir/results/log/Training_log.txt (append mode) to record best hyperparameters and validation performance per run
- Visualization (optional): matplotlib, seaborn for learning curves and result plots
- Model persistence (optional): joblib or pickle for saving trained models and artifacts
- Reproducibility practices:
  - Fixed random_state / seeds in models and CV folds to ensure consistent results across runs
- Environment management (practical note):
  - Typical use of  conda to manage dependencies


## Results

This section candidly assesses the model outcomes based on the provided visuals. All models show limited predictive power (R² values below 0.5), with notable tail errors and residual patterns. The aim is to identify where the results fall short, how the models differ, and what might be needed to make progress.

#### summary 

- None of the four models achieves strong explained variance on the test-like evaluation; all R² values are below 0.5 (RF ≈ 0.46, SVR ≈ 0.34, GPR ≈ 0.34, MLP ≈ 0.26).
- RandomForestRegressor (RF) remains the best among the four in terms of RMSE and MAE, but its performance is still far from satisfactory for high-stakes predictions.
- Tail predictions and tail-specific errors are the dominant weakness across models; central regions are somewhat better but still far from ideal.
- Stability across repeated runs is modest at best; RF shows the smallest variability, while MLP shows the largest variability among the tested models.

------

#### Figure-by-figure diagnostics (what the images reveal)

##### Figure 1: RandomForestRegressor – Predictions vs Actual

![randomforest_predictions_vs_actual](C:\Users\admin\OneDrive\Desktop\steel_production_analysis\figures\model_performances\randomforest_predictions_vs_actual.png)

- Observation: The scatter shows substantial spread around the diagonal; many points lie far from the ideal line, especially at higher Actual values.
- Interpretation: RF captures some nonlinear patterns but struggles to generalize across the full range, with noticeable under- or over-prediction in several regions.
- Takeaway: The central region looks better than the tails, but overall predictive accuracy is limited.

##### Figure 2: RandomForestRegressor – Residuals

![randomforest_residuals](C:\Users\admin\OneDrive\Desktop\steel_production_analysis\figures\model_performances\randomforest_residuals.png)

- Observation: Residuals are centered near zero but exhibit a mild pattern: increasing variability or bias at larger Actual values.
- Interpretation: Evidence of heteroscedasticity or underfitting for high targets; residual spread grows with the target.
- Takeaway: Tail variance is a key weakness for RF in this dataset.

##### Figure 3: SVR – Predictions vs Actual

![svr_predictions_vs_actual](C:\Users\admin\OneDrive\Desktop\steel_production_analysis\figures\model_performances\svr_predictions_vs_actual.png)

- Observation: Predictions align moderately with actuals in the middle range but show larger dispersion toward the tails.
- Interpretation: SVR’s tail performance is weaker; central predictions are not exceptionally accurate either.
- Takeaway: Tail behavior and overall variance indicate limited gains from SVR in this setup.

##### Figure 4: SVR – Residuals

![svr_residuals](C:\Users\admin\OneDrive\Desktop\steel_production_analysis\figures\model_performances\svr_residuals.png)

- Observation: Residuals cluster around zero in the center but drift for higher Actual values, suggesting mild systematic bias.
- Interpretation: Tail-related errors persist; kernel and parameter choices may not be sufficient to fix them.
- Takeaway: SVR stability and tail accuracy require more targeted tuning or feature engineering.

##### Figure 5: GaussianProcessRegressor – Predictions vs Actual

![gpr_predictions_vs_actual](C:\Users\admin\OneDrive\Desktop\steel_production_analysis\figures\model_performances\gpr_predictions_vs_actual.png)

- Observation: Central predictions follow the diagonal reasonably, but deviations appear at the high end of Actual.
- Interpretation: GPR provides a decent fit for the bulk of the data but tail under- or over-prediction is evident.
- Takeaway: Tail performance remains a bottleneck for GPR as well.

##### Figure 6: GaussianProcessRegressor – Residuals

![gpr_residuals](C:\Users\admin\OneDrive\Desktop\steel_production_analysis\figures\model_performances\gpr_residuals.png)

- Observation: Residuals are mostly near zero with a few outliers on the high-Actual side; some negative residuals increase as Actual grows.
- Interpretation: Overall bias is low, but tail regions exhibit higher residual variance.
- Takeaway: Tail-focused improvements are needed.

##### Figure 7: MLPRegressor – Predictions vs Actual

![mlp_predictions_vs_actual](C:\Users\admin\OneDrive\Desktop\steel_production_analysis\figures\model_performances\mlp_predictions_vs_actual.png)

- Observation: Central region shows moderate alignment; higher Actual values display systematic deviation (predictions lagging).
- Interpretation: MLP struggles to capture high-target patterns; potential underfitting or insufficient architecture.
- Takeaway: Tail accuracy is a key weakness for MLP in this case.

##### Figure 8: MLPRegressor – Residuals

![mlp_residuals](C:\Users\admin\OneDrive\Desktop\steel_production_analysis\figures\model_performances\mlp_residuals.png)

- Observation: Residuals are centered around zero with moderate spread; no strong global bias, but variability is non-negligible.
- Interpretation: Overall stability is limited; tail regions contribute to prediction errors.
- Takeaway: Architecture and regularization likely require adjustment.

##### Figure 9: Model Performance Comparison with Error Bars

![Model Performance Comparison](C:\Users\admin\OneDrive\Desktop\steel_production_analysis\figures\model_performances\Model Performance Comparison.png)

- Observation:
  - RF shows the best metrics among the four (lowest RMSE and MAE, highest R²), but all values indicate weak explanatory power.
  - SVR and GPR are similar in central performance but do not meaningfully outperform RF; MLP remains the poorest.
- Interpretation: The relative ordering is RF > SVR ≈ GPR > MLP, but none of the models achieve satisfactory explained variance.
- Takeaway: RF is the most promising baseline, but the overall performance is insufficient for reliable deployment.

------

#### Inter-model differences and what they imply

- RF remains the most stable and accurate among the tested models, but its tail performance is still deficient. This suggests RF captures some nonlinear structure but misses key patterns that drive high-target instances.
- GPR and SVR offer comparable central performance, yet both show tail weaknesses; their benefits are limited by the dataset’s characteristics and kernel/hyperparameter choices.
- MLPUnderfits: The neural network approach here does not surpass tree-based methods, likely due to insufficient data, suboptimal network architecture, or lack of effective regularization.

## Conclusion

- Overall, all four regression models show limited predictive power on the current dataset; none achieves satisfactory explained variance (R² values all below 0.5, with RF around 0.46, SVR and GPR around 0.34, and MLP around 0.26). Tail errors and residual patterns indicate weak generalization, especially for high target values.
- RandomForestRegressor (RF) is the best among the four in terms of RMSE/MAE and stability, but its tail performance remains a key weakness and it is not ready for high-stakes deployment.
- Future work should focus on data expansion and improved modeling:
  - Increase dataset size to better cover the target range.
  - Develop richer feature engineering (domain-informed interactions, material/process properties, transformations to stabilize variance).

