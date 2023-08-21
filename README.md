# CSC - Predict Returns

## Quick Start
- Clone the repo and cd into it.

    `git clone https://github.com/reagan-lopez/csc.git`

    `cd csc`

- Create python virtual environment and activate it.
    
    `python3 -m venv venv`
    
    `source venv/bin/activate`

- Install package requirements.
    
    `pip install -r requirements.txt`

- Execute [src/prediction.py](src/prediction.py) to predict the returns
    
    `python3 src/prediction.py`

    - This program trains the `Gradient Boosting Classifier` model using the provided dataset [data/train.csv](data/train.csv).
    - Subsequently, predictions are carried out on the provided dataset [data/test.csv](data/test.csv).
    - The resulting predictions are saved in the [results](results) directory. E.g. [results/predictions_20230821075213.csv](results/predictions_20230821075213.csv)


## Details

1. Conducted Exploratory Data Analysis (EDA) on the dataset [data/train.csv](data/train.csv).
   Please refer to the detailed EDA insights documented in the EDA notebook [notebook/eda.ipynb](notebook/eda.ipynb).
   - As indicated in the requirement, implemented the creation of a unique `ProductID` column based on `ProductDepartment`, `ProductCost`, and `MSRP`.
   - Calculated the `MSRP` by utilizing `DiscountPct` and `PurchasePrice` columns.
   - **Note:** The assumption was made to round the MSRP to the nearest integer, mitigating the presence of unknown ProductIDs during predictions.
   - Identified instances of `OrderID`, `ProductID` combinations with only Return records (`Returned = 1`) or coexistence of Purchase (`Returned = 0`) and Return records.
   - **Note:** The assumption was made to disregard these anomalies as part of the dataset's synthetic nature.

2. Based on the EDA insights, a preprocessing pipeline was implemented:
   - Introduced a new `MSRP` column, computed as (`PurchasePrice / (1 - DiscountPct)`), rounded to the nearest decimal (e.g., 30.0).
   - Created a `ProductID` column, distinct based on `ProductDepartment`, `ProductCost`, and `MSRP` (e.g., Youth_9_30.0).
   - Converted date columns into datetime format.
   - Derived `CustomerAge` column, indicating customer age during order, calculated as (`OrderDate - CustomerBirthDate) / 365`).
   - Excluded columns such as `ID` and `OrderDate` that offer no contribution to the target variable.
   - **Note:** The `OrderDate` would have been relevant if the orders were recent and if the minimum return day policy had been stipulated in the requirement.
   - Enabled One-Hot Encoding for categorical features like `CustomerState`, `ProductDepartment`, `ProductSize`, and `ProductID`.
   - Enabled Binning for the `CustomerAge` column using bins (0, 18, 30, 40, 50, 60, 70, inf).
   - Enabled Target Encoding on `CustomerID`, set to the mean of the `Returned` values.

3. Following thorough comparison, the **Gradient Boosting Classifier** was selected as the optimal choice, based on an assessment of scores with other classifiers. To initiate the training process, execute [src/training.py](src/training.py):
    
    `python3 src/training.py`

    - This script trains multiple classifiers and subsequently evaluates their performance using ROC AUC and Brier Score metrics.

    ```
    Evaluating model: GradientBoostingClassifier
    ROC AUC: 0.7419921284704152
    Brier Score: 0.1913358547475986

    Evaluating model: LogisticRegression
    ROC AUC: 0.7396066556626129
    Brier Score: 0.19228916873836363

    Evaluating model: AdaBoostClassifier
    ROC AUC: 0.7386169233365765
    Brier Score: 0.2398707134267459

    Evaluating model: RandomForestClassifier
    ROC AUC: 0.7137151974115621
    Brier Score: 0.19958033062197025

    ***************************************
    Best model for the dataset is:
    GradientBoostingClassifier
    ROC AUC: 0.7419921284704152
    Brier Score: 0.1913358547475986
    ***************************************
    ```
    - **NOTE** The models can be further evaluated by tuning the hyperparameters such as `n_estimators`, and `max_depth` for the ensemble classifiers and `C`, `solver`, and `max_iter` for Logistic Regression. However given the time constraint of the assignment, this step has been omitted.
    - Execute `mlflow ui` for a graphical representation of the runs and model comparisons.