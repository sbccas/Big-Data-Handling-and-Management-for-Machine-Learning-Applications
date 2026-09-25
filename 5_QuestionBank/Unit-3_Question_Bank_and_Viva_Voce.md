# 💡 Unit 3 Question Bank, Glossary & Viva Voce Exam Vault

> **Course Code:** DS-505 | **Subject:** Big Data Handling and Management for Machine Learning Applications  
> **Course Degree:** B.Sc. (Data Science & Analytics) — Semester V  
> **Institution:** Sutex Bank College of Computer Applications and Science (SBCCAS), Amroli  
> **Affiliation:** Veer Narmad South Gujarat University (VNSGU), Surat  
> **Unit Covered:** Unit 3 — Preparing Big Data for Machine Learning  
> **Document Purpose:** Comprehensive Exam Preparation, Quick Revision & Practical Viva Voce Guide  
> **Lecture Note Reference:** [Unit-3_Preparing_Big_Data_for_Machine_Learning.md](../2_Lecture_Notes/Unit-3_Preparing_Big_Data_for_Machine_Learning.md)  

---

<details open>
<summary><b>📑 Table of Contents (Click to Expand / Collapse)</b></summary>

- [1. High-Yield Technical Glossary (15 Core Definitions)](#1-high-yield-technical-glossary-15-core-definitions)
- [2. Short Answer Theory Questions (2 to 3 Marks Each)](#2-short-answer-theory-questions-2-to-3-marks-each)
- [3. Long Answer Comprehensive Theory Questions (5 to 7 Marks Each)](#3-long-answer-comprehensive-theory-questions-5-to-7-marks-each)
- [4. Practical Examination Viva Voce Questions & Model Answers](#4-practical-examination-viva-voce-questions--model-answers)

</details>

---

## 1. High-Yield Technical Glossary (15 Core Definitions)

These definitions are frequently asked in 1-mark and 2-mark university questions:

| # | Technical Term | Exact University Examination Definition |
| :-: | :--- | :--- |
| **1** | **Data Preparation** | The systematic multi-stage process of cleaning, transforming, encoding, and scaling raw datasets into refined feature matrices suitable for machine learning training. |
| **2** | **GIGO Principle** | *Garbage In, Garbage Out*; the foundational concept that the predictive accuracy of any ML model is strictly limited by the quality and cleanliness of its input training data. |
| **3** | **Data Leakage** | A fatal modeling error where information from the unseen test/validation set contaminates the training pipeline, leading to artificially inflated evaluation scores but failure in production. |
| **4** | **MCAR** | *Missing Completely at Random*; a missingness mechanism where the probability of a data point missing is completely independent of all observed and unobserved variables. |
| **5** | **Imputation** | The statistical practice of substituting missing or null values (`NaN`) in a dataset with estimated numerical or categorical proxies (such as Mean, Median, or Mode). |
| **6** | **Deduplication** | The identification and removal of redundant, repeated observations from a dataset to prevent algorithmic weighting bias and test-set contamination. |
| **7** | **Min-Max Normalization** | A feature scaling technique that linearly transforms numerical feature values into a rigid bounded interval, typically between $[0, 1]$. |
| **8** | **Z-Score Standardization** | A feature scaling technique that rescales features to have an arithmetic mean of zero ($\mu=0$) and a standard deviation of one ($\sigma=1$). |
| **9** | **Feature Engineering** | The process of selecting, combining, transforming, and creating new domain-specific variables from raw data to improve machine learning predictive performance. |
| **10** | **Label Encoding** | An encoding technique that converts categorical text labels into unique numerical integers ($0, 1, 2, \dots, K-1$), intended primarily for ordinal features or target classes. |
| **11** | **One-Hot Encoding** | An encoding technique that converts nominal categories into $K$ distinct binary indicator columns ($0$ or $1$) to avoid false mathematical ordering. |
| **12** | **Stratified Splitting** | A dataset partitioning method that preserves the exact target class label proportions identically across both the training set and testing set. |
| **13** | **Scikit-Learn Estimator** | Any Python class in Scikit-Learn that implements the `.fit()` method to learn parameters from a 2D feature matrix ($X$) and target vector ($y$). |
| **14** | **Ordinary Least Squares (OLS)**| A mathematical optimization technique used in Linear Regression that finds the optimal coefficient line by minimizing the sum of squared differences between actual and predicted values. |
| **15** | **Gini Impurity** | A mathematical measure used in Decision Trees to quantify the probability of misclassifying a randomly chosen element from a dataset node ($I_G = 1 - \sum p_i^2$). |

---

## 2. Short Answer Theory Questions (2 to 3 Marks Each)

#### Q1: What is Data Leakage, and why is it dangerous in applied Machine Learning?
> **Model Answer:**  
> **Data Leakage** occurs when information from outside the training dataset (specifically from the test or validation set) is accidentally used to create or tune the model.  
> * **Danger:** The model learns features or distribution parameters (such as the global mean or test-set minimum/maximum) that it would not have in real-world deployment. Consequently, the model achieves near-perfect test scores in lab notebooks, but performs terribly when deployed to production.

#### Q2: Compare Mean Imputation and Median Imputation. When should Median be preferred?
> **Model Answer:**  
> * **Mean Imputation:** Replaces missing values with the arithmetic average. *Limitation:* It is heavily distorted by extreme outliers and skews the natural variance.  
> * **Median Imputation:** Replaces missing values with the 50th percentile (middle value).  
> * **When to prefer Median:** Median must always be preferred for **skewed numerical features** containing extreme outliers (e.g., Annual Salaries, House Prices, or Webpage Visit Durations), because the median is mathematically resistant to outlier corruption.

#### Q3: Differentiate between Min-Max Scaling and Z-Score Standardization.
> **Model Answer:**  
> * **Min-Max Scaling (`MinMaxScaler`):** Compresses values into a fixed bounded range $[0, 1]$ using formula $X_{\text{norm}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$. It is highly sensitive to extreme outliers.  
> * **Z-Score Standardization (`StandardScaler`):** Rescales data to center at $\mu=0$ with $\sigma=1$ using formula $z = \frac{X - \mu}{\sigma}$. It has unbounded output and handles outliers gracefully without squashing normal observations.

#### Q4: What is the Dummy Variable Trap in One-Hot Encoding and how do we prevent it?
> **Model Answer:**  
> * **The Trap:** When a categorical feature with $K$ categories is converted into $K$ binary columns, the columns exhibit perfect **multicollinearity** (one column can be perfectly predicted from the sum of the remaining $K-1$ columns). This destabilizes matrix inversion in regression models.  
> * **Prevention:** Drop one reference column by setting `drop="first"` in Scikit-Learn’s `OneHotEncoder` or `drop_first=True` in Pandas `pd.get_dummies()`.

#### Q5: Why is K-Nearest Neighbors (KNN) referred to as a "Lazy Learner"?
> **Model Answer:**  
> KNN is called a **Lazy Learner (Instance-Based Learner)** because its `.fit()` method does not learn an explicit mathematical model or adjust weights. It simply stores the training instances in memory. All heavy computations (calculating geometric distances to find the $K$ closest neighbors) are deferred until `.predict()` is called during inference.

---

## 3. Long Answer Comprehensive Theory Questions (5 to 7 Marks Each)

#### Q1: Explain the Complete Machine Learning Data Preparation Pipeline with a neat architectural diagram.
> **Structuring Your Answer for Full Marks:**
> 1. **Introduction:** Define data preparation and explain the GIGO principle.
> 2. **Pipeline Architecture Diagram:** Draw the 5-stage flowchart: Raw Dirty Big Data $\rightarrow$ Data Cleaning $\rightarrow$ Feature Engineering $\rightarrow$ Stratified Dataset Splitting $\rightarrow$ Model Training & Diagnostic Evaluation.
> 3. **Data Cleaning Phase:** Missing value identification (`isna()`), statistical imputation (median/mode), and deduplication (`drop_duplicates()`).
> 4. **Feature Engineering Phase:** Categorical encoding (One-Hot vs. Label), feature selection via correlation, and numerical scaling (`StandardScaler`).
> 5. **Dataset Partitioning:** Explain why splitting must occur before fitting transformers to prevent Data Leakage.
> 6. **Model Execution:** Universal Scikit-Learn interface (`.fit()` and `.predict()`).

#### Q2: Compare Normalization (`MinMaxScaler`) and Standardization (`StandardScaler`) in Detail with Mathematical Formulas, Outlier Impacts, and Algorithm Requirements.
> **Structuring Your Answer for Full Marks:**
> 1. **Necessity of Scaling:** Explain why distance-based algorithms (KNN, SVM, K-Means) and gradient descent models fail without comparable feature scales.
> 2. **Min-Max Normalization:**
>    * Mathematical formula: $X_{\text{norm}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$.
>    * Properties: Strictly bounded $[0, 1]$, preserves original shape, catastrophic distortion in the presence of outliers.
> 3. **Z-Score Standardization:**
>    * Mathematical formula: $z = \frac{X - \mu}{\sigma}$.
>    * Properties: Centered at 0, unit standard deviation, handles outliers without compressing normal data.
> 4. **Algorithm Sensitivity:** Detail why KNN and Linear Regression require scaling, while Decision Trees are completely scale-invariant.
> 5. **Comparative Table:** Construct a 5-row table comparing formula, bounds, outlier sensitivity, and primary use cases.

#### Q3: Explain Categorical Encoding Techniques (One-Hot vs. Label Encoding) and Dataset Partitioning (Random vs. Stratified Splitting).
> **Structuring Your Answer for Full Marks:**
> 1. **Categorical Nature:** Differentiate between Ordinal (inherent rank) and Nominal (no rank) variables.
> 2. **Label Encoding:**
>    * Explain integer mapping ($0, 1, \dots, K-1$).
>    * Detail why Label Encoding nominal features creates bogus mathematical rankings ($\text{City A} < \text{City B}$).
> 3. **One-Hot Encoding:**
>    * Explain binary vector expansion.
>    * Explain the Dummy Variable Trap and the mathematical need for `drop="first"`.
> 4. **Dataset Splitting:**
>    * Define training set (learning parameters) versus testing set (generalization evaluation).
>    * Contrast Random Splitting with Stratified Splitting (`stratify=y`).
>    * Explain why Stratification is mandatory for imbalanced datasets (e.g., 98% non-churn, 2% churn).

#### Q4: Detail the Working Principles, Mathematical Formulations, and Evaluation Metrics of Linear Regression, Decision Tree Classifier, and K-Nearest Neighbors.
> **Structuring Your Answer for Full Marks:**
> 1. **Linear Regression:**
>    * Formula: $\hat{y} = \beta_0 + \sum \beta_i X_i$.
>    * OLS cost function: Residual Sum of Squares (RSS).
>    * Metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and $R^2$ Score.
> 2. **Decision Tree Classifier:**
>    * Recursive binary splitting and tree structure (Root, Node, Leaf).
>    * Splitting criteria: Gini Impurity ($1 - \sum p_i^2$) vs. Entropy ($-\sum p_i \log_2 p_i$).
>    * Regularization hyperparameters: `max_depth`, `min_samples_split`.
> 3. **K-Nearest Neighbors Classifier:**
>    * Instance-based lazy learning principle.
>    * Euclidean distance equation in $n$-dimensional space.
>    * Majority voting mechanism, impact of $K$ (overfitting vs. underfitting), and mandatory feature scaling.

---

## 4. Practical Examination Viva Voce Questions & Model Answers

Examiners frequently ask these 15 questions during university practical examinations:

#### 1. What is the difference between `.fit()`, `.transform()`, and `.fit_transform()` in Scikit-Learn?
> **Answer:**  
> * `.fit()` computes and learns the internal parameters (e.g., mean and standard deviation in `StandardScaler`, or classes in `LabelEncoder`).  
> * `.transform()` applies the learned parameters to scale or encode a dataset.  
> * `.fit_transform()` is a convenient optimization that computes parameters and transforms the training dataset in a single pass.

#### 2. Can you call `.fit_transform()` on the test dataset? Why or why not?
> **Answer:** **NO! Never call `.fit_transform()` on the test set.** Doing so re-calculates parameters from the test set, causing **Data Leakage**. You must call `.fit_transform()` strictly on the training set, and only call `.transform()` on the test set.

#### 3. Why is feature scaling mandatory for K-Nearest Neighbors (KNN)? What happens if you skip it?
> **Answer:** KNN calculates Euclidean geometric distance: $d = \sqrt{\sum (x_1 - x_2)^2}$. If one feature has large numbers (e.g., Income in thousands) and another has small numbers (e.g., Age in tens), the Income feature will dominate 99.9% of the distance calculation, making the Age feature completely irrelevant.

#### 4. Does a Decision Tree require feature scaling? Why or why not?
> **Answer:** **No.** Decision Trees are scale-invariant. A tree splits data by asking binary monotonic questions on individual features independently (e.g., `Income <= 50,000`). Multiplying or shifting feature values does not alter the ordering or the purity of the splits.

#### 5. What is the purpose of the `random_state` parameter in `train_test_split()`?
> **Answer:** It sets the seed for the pseudo-random number generator, ensuring that the random shuffle and split produces the exact same data partitions every time the code is executed, making laboratory experiments reproducible.

#### 6. What does `stratify=y` do in `train_test_split()`?
> **Answer:** It ensures that the percentage of each target class in the training split matches the percentage in the testing split. This is essential for imbalanced classification tasks (e.g., rare disease diagnosis or credit fraud).

#### 7. How do you detect missing values in a Pandas DataFrame?
> **Answer:** Using `df.isna().sum()` or `df.isnull().sum()`, which returns the count of missing (`NaN`) entries for each column in the DataFrame.

#### 8. What is the difference between `SimpleImputer(strategy="mean")` and `SimpleImputer(strategy="median")`?
> **Answer:** `"mean"` replaces missing values with the arithmetic average, which is sensitive to outliers. `"median"` replaces missing values with the 50th percentile, which is robust against extreme outliers.

#### 9. What is the difference between an Ordinal feature and a Nominal feature?
> **Answer:** An **Ordinal feature** has a natural, meaningful mathematical ranking (e.g., Low, Medium, High; or High School, Bachelor, Master). A **Nominal feature** has no intrinsic ordering (e.g., Colors: Red, Green, Blue; or Cities: Surat, Mumbai).

#### 10. How does `OneHotEncoder` handle unseen categories during inference if `handle_unknown="ignore"` is set?
> **Answer:** If a new category appears in the test data that was never seen during training, `OneHotEncoder` encodes it as all zeros across the binary columns instead of throwing a fatal runtime error.

#### 11. What is the Residual Sum of Squares (RSS) in Linear Regression?
> **Answer:** It is the sum of squared differences between the actual observed values ($y_i$) and the predicted values ($\hat{y}_i$): $\text{RSS} = \sum (y_i - \hat{y}_i)^2$. Linear Regression minimizes this metric.

#### 12. What does an $R^2$ score of $0.85$ indicate?
> **Answer:** It indicates that **85% of the total variance** in the dependent target variable ($y$) is successfully explained by the independent features in the linear regression model.

#### 13. How do you prevent a Decision Tree from overfitting?
> **Answer:** By setting regularizing tree-pruning hyperparameters: restricting `max_depth` (e.g., 3 to 5), increasing `min_samples_split`, or increasing `min_samples_leaf`.

#### 14. What distance metric does `KNeighborsClassifier` use by default?
> **Answer:** It uses the **Minkowski metric** with $p=2$, which is mathematically equivalent to the standard **Euclidean distance**.

#### 15. What is the difference between Euclidean distance and Manhattan distance?
> **Answer:**  
> * **Euclidean Distance ($L_2$ norm):** The straight-line distance between two points: $\sqrt{\sum (p_i - q_i)^2}$.  
> * **Manhattan Distance ($L_1$ norm / City Block):** The grid-like distance traversing right angles: $\sum |p_i - q_i|$.

---
*(End of Unit 3 Question Bank & Viva Voce Exam Vault)*
