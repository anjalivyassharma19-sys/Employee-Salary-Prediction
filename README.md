# Employee-Salary-Prediction
Employee salary prediction using machine learning and employee data.
# Employee Data Preprocessing & Salary Prediction

## Overview

This project demonstrates an end-to-end **employee data preprocessing
and regression workflow using Python**. The work starts with an employee
dataset stored in an Excel file and covers data inspection,
missing-value handling, duplicate detection, outlier treatment, feature
engineering, categorical encoding, polynomial feature generation,
train/test splitting, linear regression, and model evaluation.

The original dataset contains **62 rows and 13 columns** and includes
employee information such as age, gender, city, department, education
level, years of experience, join date, remote-work status, completed
projects, performance score, and salary.

## Dataset

The input dataset is:

``` text
employee_preprocessing_dataset.xlsx
```

### Main columns

  Column                 Description
  ---------------------- --------------------------------------
  `Employee_ID`          Employee identifier
  `Name`                 Employee name
  `Age`                  Employee age
  `Gender`               Gender category
  `City`                 Employee city
  `Department`           Department
  `Education_Level`      Education category
  `Years_Experience`     Years of professional experience
  `Join_Date`            Employee joining date
  `Remote_Work`          Whether the employee works remotely
  `Projects_Completed`   Number of completed projects
  `Performance_Score`    Employee performance score
  `Salary`               Salary used as the regression target

## Project Workflow

The analysis follows these main steps:

1.  Load the Excel dataset with Pandas.
2.  Inspect the dataset shape and data types.
3.  Generate descriptive statistics.
4.  Identify missing values.
5.  Detect and remove duplicate rows.
6.  Experiment with different missing-value treatment methods.
7.  Convert and process date information.
8.  Detect outliers using IQR and Z-score methods.
9.  Create a `yrs_company` feature from `Join_Date`.
10. Encode categorical data, including `Education_Level`.
11. Select numerical features for modelling.
12. Generate polynomial features.
13. Split the data into training and testing sets.
14. Train a `LinearRegression` model.
15. Generate salary predictions.
16. Evaluate the model using RMSE.
17. Visualise the relationship between experience and salary.

## Data Quality Checks

### Missing values

The initial dataset contains missing values in several columns. The
notebook checks missing-value counts using:

``` python
data.isnull().sum()
```

The initial missing-value counts include:

-   `Age`: 4
-   `Gender`: 4
-   `City`: 2
-   `Education_Level`: 3
-   `Years_Experience`: 5
-   `Remote_Work`: 2
-   `Projects_Completed`: 2
-   `Performance_Score`: 6
-   `Salary`: 5

### Duplicate records

Duplicate records are identified with:

``` python
data.duplicated().sum()
```

The original dataset contains **2 duplicate rows**. They are removed
using:

``` python
data = data.drop_duplicates()
```

After removal, the duplicate count is reported as zero.

## Missing-Value Handling

The project explores several approaches for handling missing values,
including:

``` python
data.fillna(0)
```

Mean-based replacement:

``` python
data["Age"] = data["Age"].fillna(data["Age"].mean())
```

Median replacement:

``` python
data["Salary"] = data["Salary"].fillna(data["Salary"].median())
```

Mode replacement:

``` python
data["Salary"] = data["Salary"].fillna(data["Salary"].mode()[0])
```

Forward fill and backward fill are also explored for categorical data
such as `Gender`:

``` python
data["Gender"] = data["Gender"].ffill()
data["Gender"] = data["Gender"].bfill()
```

Interpolation is demonstrated for numeric data:

``` python
data["Age"] = data["Age"].interpolate()
```

> **Note:** These approaches are demonstrated as part of the learning
> workflow. In a production pipeline, the appropriate strategy should be
> selected based on the meaning and distribution of each column rather
> than applying a single method to every variable.

## Outlier Detection

The project uses both **IQR** and **Z-score** approaches.

### IQR method

For age, the first and third quartiles are calculated and the
interquartile range is used to establish lower and upper limits:

``` python
Q1 = df_age["Age"].quantile(0.25)
Q3 = df_age["Age"].quantile(0.75)

IQR = Q3 - Q1

Lower_Limit = Q1 - (1.5 * IQR)
Upper_Limit = Q3 + (1.5 * IQR)
```

The recorded age limits in the analysis are:

``` text
Lower limit: 3.75
Upper limit: 54.75
```

Rows outside these limits are filtered.

### Z-score method

The project also uses SciPy's Z-score calculation for variables such as:

-   `Years_Experience`
-   `Salary`
-   `Age`

Example:

``` python
z_score = np.abs(stats.zscore(data["Salary"]))
data = data[z_score <= 3]
```

## Feature Engineering

The project creates a new feature called:

``` text
yrs_company
```

It is calculated from the employee's `Join_Date`:

``` python
data["Join_Date"] = pd.to_datetime(data["Join_Date"])

data["yrs_company"] = (
    pd.Timestamp.today() - data["Join_Date"]
).dt.days / 365.25
```

This converts the joining date into an approximate number of years at
the company.

## Categorical Encoding

`Education_Level` is prepared for machine-learning use using one-hot
encoding.

The workflow explores `OneHotEncoder` from scikit-learn and creates
encoded columns for education categories.

The original `Education_Level` column is then removed before continuing
with the numerical modelling dataset.

## Polynomial Features

The modelling workflow creates polynomial features from selected
numerical variables using scikit-learn's `PolynomialFeatures`.

The features used include:

``` text
Years_Experience
Performance_Score
yrs_company
Projects_Completed
```

A degree-3 polynomial transformation is explored:

``` python
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(poly_data)
```

## Model Training

The regression target is:

``` text
Salary
```

The input features are:

``` python
X = data[
    [
        "Years_Experience",
        "Performance_Score",
        "yrs_company",
        "Projects_Completed"
    ]
]

Y = data["Salary"]
```

The data is split into training and testing sets using:

``` python
train_test_split(X, Y, test_size=0.2)
```

A `LinearRegression` model is then trained:

``` python
from sklearn.linear_model import LinearRegression

Model = LinearRegression()
Model.fit(X_poly_train, Y_train)
```

## Prediction

The trained model is used to generate salary predictions:

``` python
Y_pred = Model.predict(X_poly_test)
```

The original analysis also demonstrates predicting salary for a sample
feature set.

## Model Evaluation

The project evaluates the regression model using **Root Mean Squared
Error (RMSE)**:

``` python
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
```

The recorded RMSE from the analysis is approximately:

``` text
RMSE: 25,165.47
```

The analysis also calculates RMSE as a percentage of the mean test
salary:

``` text
RMSE percentage: approximately 26.26%
```

These figures describe the particular preprocessing and train/test split
used in the source work and should not be interpreted as a general
measure of model performance on new datasets.

## Visualisation

The project includes exploratory visualisation of salary against years
of experience:

``` python
plt.scatter(data["Years_Experience"], data["Salary"])

plt.xlabel("Yrs_Exp")
plt.ylabel("Salary")
plt.title("Growth In Salary")

plt.show()
```

This is intended to help explore the relationship between experience and
salary.

## Technologies Used

-   **Python**
-   **Pandas** --- data loading and manipulation
-   **NumPy** --- numerical operations
-   **SciPy** --- statistical calculations and Z-scores
-   **Matplotlib** --- visualisation
-   **Scikit-learn** --- encoding, feature engineering, train/test
    splitting, regression, and evaluation
-   **Excel** --- source dataset

## Example Imports

``` python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
```

## Suggested Project Structure

``` text
employee-data-preprocessing/
│
├── employee_preprocessing_dataset.xlsx
├── employee_preprocessing.py
├── README.md
└── requirements.txt
```

## Installation

Create a Python environment and install the required packages:

``` bash
pip install pandas numpy scipy matplotlib scikit-learn openpyxl
```

## Running the Project

1.  Place `employee_preprocessing_dataset.xlsx` in the project
    directory.
2.  Open the Python script or notebook containing the preprocessing
    workflow.
3.  Run the data-loading and preprocessing steps.
4.  Run the feature-engineering and encoding steps.
5.  Train the regression model.
6.  Review the predictions and RMSE evaluation.

## Important Notes

The supplied analysis is a learning/exploration workflow and contains
several intermediate experiments and error outputs. For example, the
source shows older scikit-learn method names such as
`get_feature_names`, which produced `AttributeError`s, and it also
contains an intermediate `fillna(data["Salary"].mean)` expression that
is missing the method call.

For a clean production implementation, these steps should be
consolidated into a reproducible preprocessing pipeline, with missing
values handled before modelling and transformations fitted only on the
training data.

## Learning Outcomes

This project demonstrates practical understanding of:

-   Data inspection
-   Data quality assessment
-   Missing-data handling
-   Duplicate removal
-   Outlier detection
-   Feature engineering
-   Categorical encoding
-   Polynomial feature generation
-   Train/test splitting
-   Linear regression
-   Regression evaluation using RMSE
-   Basic data visualisation
