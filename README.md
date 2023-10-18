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

    - This program trains the `Random Forest Classifier` model using the provided dataset [artifacts/training_data.csv](artifacts/training_data.csv).
    - Subsequently, predictions are carried out on the provided dataset [artifacts/prediction_data.csv](artifacts/prediction_data.csv).
    - The resulting predictions are saved in the [results](results) directory. E.g. [results/predictions_20231018051908.csv](results/predictions_20231018051908.csv)


## Details

1. Conducted Exploratory Data Analysis (EDA) on the dataset [artifacts/training_data.csv](artifacts/training_data.csv).
   Please refer to the detailed EDA insights documented in the EDA notebook [notebook/eda.ipynb](notebook/eda.ipynb).
   - As indicated in the requirement, implemented the creation of a unique `ProductID` column based on `ProductDepartment`, `ProductCost`, and `MSRP`.
   - Calculated the `MSRP` by utilizing `DiscountPct` and `PurchasePrice` columns.
   - **Note:** The assumption was made to round the MSRP to the nearest integer, mitigating the presence of unknown ProductIDs during predictions.
   - Identified instances of `OrderID`, `ProductID` combinations with only Return records (`Returned = 1`) or coexistence of Purchase (`Returned = 0`) and Return records.

2. Based on the EDA insights, a preprocessing pipeline was implemented:
   - Introduced a new `MSRP` column, computed as (`PurchasePrice / (1 - DiscountPct)`), rounded to the nearest decimal (e.g., 30.0).
   - Created a `ProductID` column, distinct based on `ProductDepartment`, `ProductCost`, and `MSRP` (e.g., Youth-9-30.0).
   - Removed original purchase records (`Returned = 0`) for `OrderID`, `ProductID` combinations that have already been returned (`Returned = 1`).
   - Converted date columns into datetime format.
   - Derived `CustomerAge` column, indicating customer age during order, calculated as (`OrderDate - CustomerBirthDate) / 365`).
   - Excluded columns such as `ID` and `OrderDate` that offer no contribution to the target variable. Also excluded `CustomerID`.
   - **Note:** The `OrderDate` would have been relevant if the orders were recent and if the minimum return day policy had been stipulated in the requirement.
   - Enabled One-Hot Encoding for categorical features like `CustomerState`, `ProductDepartment`, `ProductSize`, and `ProductID`.

3. Following thorough comparison, the **Random Forest Classifier** was selected as the optimal choice, based on an assessment of scores with other classifiers. To initiate the training process, execute [src/training.py](src/training.py):
    
    `python3 src/training.py`

    - This script trains multiple classifiers and subsequently evaluates their performance using ROC AUC and Brier Score metrics.

    ```
    Evaluating model: GradientBoostingClassifier
    ROC AUC: 0.6080930018716165
    Brier Score: 0.22552069527411897

    Evaluating model: LogisticRegression
    ROC AUC: 0.6015078525716959
    Brier Score: 0.22602833099129133

    Evaluating model: AdaBoostClassifier
    ROC AUC: 0.6039288770892809
    Brier Score: 0.24888510298349534

    Evaluating model: RandomForestClassifier
    ROC AUC: 0.6738378005099563
    Brier Score: 0.21208089565738358

    ***************************************
    Best model for the dataset is:
    RandomForestClassifier
    ROC AUC: 0.6738378005099563
    Brier Score: 0.21208089565738358
    ***************************************
    ```
    - **NOTE** The models can be further evaluated by tuning the hyperparameters such as `n_estimators`, and `max_depth` for the ensemble classifiers and `C`, `solver`, and `max_iter` for Logistic Regression. However given the time constraint of the assignment, this step has been omitted.
    - Execute `mlflow ui` for a graphical representation of the runs and model comparisons.