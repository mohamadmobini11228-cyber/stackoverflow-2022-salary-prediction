# Stack Overflow 2022 Salary Prediction

Predicting developer yearly compensation from the Stack Overflow 2022 Developer Survey using linear regression.

## Overview

This project explores how factors like experience, country, education, remote-work status, and tech stack relate to developer salary, then trains a regression model to predict yearly compensation (`ConvertedCompYearly`) from these features.

- **Dataset:** Stack Overflow 2022 Developer Survey (`survey_results_public.csv`), ~70K respondents
- **Model:** Linear Regression
- **Result:** R² = 0.56 on held-out test data

## Approach

**1. Exploratory analysis**
Looked at the distribution of respondents by country, employment type, and other categorical fields, and checked for missing data across columns.

**2. Outlier handling**
Yearly compensation had a long right tail (a small number of extreme values stretching into the tens of millions), which would have distorted a linear model. Trimmed the target to the 1st–99th percentile before training.

<img width="954" height="448" alt="compensation_outliers_boxplot (1)" src="https://github.com/user-attachments/assets/085455ed-92b8-407c-a047-3ebf17c87a59" />

**3. Target transform**
Applied a log transform (`log1p`) to `ConvertedCompYearly` to reduce skew and make the relationship with input features closer to linear.

**4. Feature engineering**
- Grouped rare categories (fewer than 20 occurrences) into an `"Other"` bucket to reduce noise from low-frequency values
- One-hot encoded single-valued categorical fields (education level, age bracket, org size, etc.)
- Multi-label encoded fields where respondents could select multiple values (e.g. languages, frameworks, tools, databases) by splitting on `;` and creating indicator columns
- Target-encoded high-cardinality fields (e.g. Country) with smoothing to avoid overfitting on rare categories
- Mapped ordinal text fields (e.g. remote-work status, years of experience) to numeric scales

**5. Missing values & scaling**
Imputed remaining missing values with the most frequent value per column, then standardized all features before training.

**6. Train/test split & evaluation**
Split data 80/20 and evaluated with R² on both sets to check for overfitting.

## Tech Stack
Python, Pandas, NumPy, Scikit-Learn, category_encoders, Matplotlib, Seaborn, Plotly

## What I'd improve next
- Try a non-linear model (XGBoost/Random Forest) to capture interactions a linear model misses
- Add cross-validation instead of a single train/test split
- Investigate feature importance to understand which factors drive compensation most
