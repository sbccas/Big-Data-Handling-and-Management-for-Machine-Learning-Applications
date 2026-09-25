<div align="center">

# SUTEX BANK COLLEGE OF COMPUTER APPLICATIONS & SCIENCE, AMROLI
### T.Y.B.Sc. (DATA SCIENCE & ANALYTICS) — SEMESTER: 5th
## ASSIGNMENT – III (UNIT – III)
### SUBJECT: DS-505 — Big Data Handling and Management for Machine Learning Applications

**Unit Covered:** Unit 3 – Preparing Big Data for Machine Learning  
**Submission Date:** *To be announced by Course Instructor*  
**Recommended Student Notebook Length:** **12 to 15 Pages Total**  

---

</div>

> **Instructions for Students:**
> 1. Handwrite all answers neatly in your official course assignment notebook / sheets.
> 2. Each **Long Question** should take **~2 to 2.5 pages** (including architecture diagrams, mathematical derivations, and comparison tables).
> 3. Each **Short Question** should take **~0.5 to 0.75 page**.
> 4. For **Multiple Choice Questions (MCQs)**, write the question number, correct option letter, and full option text.
> 5. Total assignment volume should be between **12 to 15 handwritten pages**.
> 6. All questions are directly searchable and referenced from your lecture notes:  
>    👉 [**`2_Lecture_Notes/Unit-3_Preparing_Big_Data_for_Machine_Learning.md`**](../2_Lecture_Notes/Unit-3_Preparing_Big_Data_for_Machine_Learning.md)

---

## SECTION I: Answer in Detail (Long Questions)
*(Attempt all questions. Expected length: ~2 to 2.5 handwritten pages per question)*

### 1. Explain the Data Preparation Lifecycle, the GIGO Principle, and the Prevention of Data Leakage in Machine Learning.
* Discuss why 70% to 80% of total machine learning project time is dedicated to data preparation.
* Detail the **Garbage In, Garbage Out (GIGO)** principle with real-world Big Data examples.
* Define **Data Leakage (Train-Test Contamination)**: Explain how fitting a scaler or imputer on the full dataset before splitting leads to false high accuracy and production failure.
* State the **Golden Rule of Preprocessing**: Why must `.fit()` be executed strictly on `X_train` while `X_test` is processed via `.transform()`?
* Draw the complete end-to-end Machine Learning data preparation architecture diagram.  
*(📖 **Reference:** Lecture Notes `Section 1.1`, `Section 1.2`, `Section 1.3`, & `Section 1.4`)*

---

### 2. Explain Missing Data Mechanisms (MCAR, MAR, MNAR) and Detail All Major Missing Value Treatment Strategies.
* Define and contrast the three statistical mechanisms of missing data:
  1. **MCAR (Missing Completely at Random)**
  2. **MAR (Missing at Random)**
  3. **MNAR (Missing Not at Random)**
* Compare the two overarching remediation strategies:
  * **Deletion:** Listwise row removal (`dropna(axis=0)`) vs. Columnar feature dropping (`dropna(axis=1)`). When is deletion acceptable and when does it introduce selection bias?
  * **Statistical Imputation:** Univariate Mean vs. Median vs. Mode imputation. Explain why the **Median** is superior to the Mean for skewed features (e.g., salaries).
* Provide a clean Python code example using Scikit-Learn’s `SimpleImputer` for both numeric and categorical columns.  
*(📖 **Reference:** Lecture Notes `Section 2.1`)*

---

### 3. Differentiate between Data Normalization (Min-Max Scaling) and Data Standardization (Z-Score) with Mathematical Derivations and Use Cases.
* Why do machine learning algorithms require numerical feature scaling?
* **Min-Max Normalization:**
  * Write the mathematical formula: $X_{\text{norm}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$.
  * State the bounded output range ($[0, 1]$) and discuss its extreme vulnerability to outliers.
* **Z-Score Standardization (`StandardScaler`):**
  * Write the mathematical formula: $z = \frac{X - \mu}{\sigma}$.
  * Define the population mean ($\mu$) and standard deviation ($\sigma$).
  * Explain why Standardization is more resilient to outliers than Min-Max scaling.
* Construct a detailed comparative table comparing both techniques across formula, output bounds, outlier sensitivity, and algorithm use cases.
* Draw the visualization diagrams comparing bounded Min-Max boxes against Gaussian bell curves.  
*(📖 **Reference:** Lecture Notes `Section 2.3`)*

---

### 4. Detail Categorical Data Encoding (One-Hot vs. Label Encoding) and Dataset Partitioning (Random vs. Stratified Splitting).
* Why can machine learning algorithms not process raw text categorical features directly?
* Differentiate between **Ordinal** (ordered) and **Nominal** (unordered) categorical variables.
* **Label Encoding (`LabelEncoder`):**
  * Explain the integer mapping mechanism.
  * Why does using Label Encoding on nominal variables (e.g., cities) introduce false mathematical ranking?
* **One-Hot Encoding (`OneHotEncoder`):**
  * Explain binary column generation.
  * Explain the **Dummy Variable Trap (Multicollinearity)** and how `drop="first"` resolves it.
* **Dataset Splitting:**
  * Compare **Random Splitting** versus **Stratified Splitting (`stratify=y`)**.
  * Why is Stratification mandatory for imbalanced target classification (e.g., fraud or churn)?  
*(📖 **Reference:** Lecture Notes `Section 3.2` & `Section 3.3`)*

---

### 5. Detail the Three Fundamental Machine Learning Models: Linear Regression, Decision Tree Classifier, and K-Nearest Neighbors (KNN).
* For each of the three mandatory syllabus models, explain:
  1. **Linear Regression (`LinearRegression`):**
     * Mathematical equation: $\hat{y} = \beta_0 + \sum \beta_i X_i$.
     * Ordinary Least Squares (OLS) minimization of Residual Sum of Squares (RSS).
     * Evaluation metrics: Mean Squared Error (MSE) and $R^2$ Score.
  2. **Decision Tree Classifier (`DecisionTreeClassifier`):**
     * Tree anatomy: Root node, internal decision splits, and leaf nodes.
     * Mathematical splitting criteria: **Gini Impurity** vs. **Entropy / Information Gain**.
     * Overfitting control: The role of `max_depth` and `min_samples_split`.
  3. **K-Nearest Neighbors Classifier (`KNeighborsClassifier`):**
     * Concept of non-parametric "Lazy Learning" (no explicit training phase).
     * Euclidean distance calculation in $n$-dimensional space.
     * The role of hyperparameter $K$ (low $K$ overfitting vs. high $K$ underfitting) and why feature scaling is strictly mandatory.
* Draw the comparative visual diagrams illustrating all three models.  
*(📖 **Reference:** Lecture Notes `Section 6.1` to `Section 6.4`)*

---

## SECTION II: Answer Briefly (Short Questions)
*(Attempt all questions. Expected length: ~0.5 to 0.75 handwritten page per question)*

### 1. Differentiate between `.fit()`, `.transform()`, and `.fit_transform()` in Scikit-Learn.
* Explain the parameter calculation phase versus the dataset transformation phase.  
*(📖 **Reference:** Lecture Notes `Section 4.1` & `Section 5.4`)*

---

### 2. Why is Feature Scaling Mandatory for K-Nearest Neighbors (KNN) but Optional for Decision Trees?
* Contrast geometric distance metrics with hierarchical orthogonal binary threshold splits.  
*(📖 **Reference:** Lecture Notes `Section 2.3` & `Section 6.4`)*

---

### 3. What is the Dummy Variable Trap in One-Hot Encoding and How is it Mitigated?
* Explain perfect multicollinearity between $K$ binary columns and the role of `drop='first'`.  
*(📖 **Reference:** Lecture Notes `Section 3.2`)*

---

### 4. Differentiate between Gini Impurity and Information Gain (Entropy) in Decision Trees.
* Write down the mathematical formulas and compare their computational efficiency during node splitting.  
*(📖 **Reference:** Lecture Notes `Section 6.3`)*

---

### 5. Why is Removing Duplicate Records Critical Before Splitting Data into Training and Testing Sets?
* Explain the risk of algorithm memorization and artificial inflation of test accuracy scores.  
*(📖 **Reference:** Lecture Notes `Section 2.2`)*

---

## SECTION III: Multiple Choice Questions (MCQs)
*(Choose the single correct option. Write the question number, option letter, and answer text).*

#### 1. In Scikit-Learn, which method is used by an Estimator to learn parameters (e.g., mean, variance, weights) from training data?
- **A)** `.predict()`
- **B)** `.transform()`
- **C)** `.fit()`
- **D)** `.score()`

#### 2. What is the mathematical output range of features normalized using Min-Max Scaling?
- **A)** $[-1.0, +1.0]$
- **B)** $[0.0, 1.0]$
- **C)** $[-\infty, +\infty]$
- **D)** $[0.0, 100.0]$

#### 3. If a dataset has an imbalanced binary classification target (95% No, 5% Yes), which parameter in `train_test_split()` ensures identical class ratios in both splits?
- **A)** `shuffle=True`
- **B)** `random_state=42`
- **C)** `stratify=y`
- **D)** `test_size=0.05`

#### 4. Applying `StandardScaler` to a feature transforms its distribution to have:
- **A)** $\text{Mean} = 1.0$ and $\text{Variance} = 0.0$
- **B)** $\text{Mean} = 0.0$ and $\text{Standard Deviation} = 1.0$
- **C)** $\text{Minimum} = 0.0$ and $\text{Maximum} = 1.0$
- **D)** $\text{Median} = 0.0$ and $\text{IQR} = 1.0$

#### 5. Why is K-Nearest Neighbors (KNN) classified as a "Lazy Learner"?
- **A)** It takes very long to compute predictions
- **B)** It does not learn an explicit discriminative function during the `.fit()` stage
- **C)** It only processes binary categorical features
- **D)** It can only classify linearly separable data

#### 6. In a Decision Tree, what is the Gini Impurity of a completely pure leaf node (where all samples belong to one class)?
- **A)** $1.0$
- **B)** $0.5$
- **C)** $0.0$
- **D)** $-1.0$

#### 7. What fatal problem occurs if a data scientist fits `StandardScaler` on the entire dataset before calling `train_test_split()`?
- **A)** Vanishing Gradient Problem
- **B)** Data Leakage (Train-Test Contamination)
- **C)** High Bias / Underfitting
- **D)** Integer Overflow Error

---

### 🔑 Answer Key for Section III (MCQs)

| Question # | Correct Option | Brief Technical Justification |
| :---: | :---: | :--- |
| **1** | **C) `.fit()`** | `.fit()` calculates and stores internal parameters (such as `mean_` or regression weights) in the estimator object. |
| **2** | **B) $[0.0, 1.0]$** | Min-Max scaling compresses values strictly between the minimum ($0$) and maximum ($1$) bounds. |
| **3** | **C) `stratify=y`** | Stratification forces the train-test split to sample observations proportionally across target class labels. |
| **4** | **B) $\text{Mean} = 0.0, \text{Std} = 1.0$** | Z-score standardization centers data at zero and scales it to unit standard deviation. |
| **5** | **B) No explicit discriminative function** | KNN merely stores the training points; all geometric proximity calculations are deferred until inference. |
| **6** | **C) $0.0$** | When all samples belong to class 1, $p_1 = 1$, and Gini impurity $= 1 - (1)^2 = 0.0$ (maximum purity). |
| **7** | **B) Data Leakage** | The scaler absorbs global distribution information from the test set, contaminating the model's training process. |

---

<br>

<div align="right">

**Asst. Prof. Hitesh Patel**  
*Department of Computer Science & Data Science*  
*Sutex Bank College of Computer Applications & Science (SBCCAS), Amroli*  
*F.Y. / T.Y. B.Sc. (Data Science & Analytics) & B.C.A.*  

</div>
