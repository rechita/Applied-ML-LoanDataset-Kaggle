# Applied-ML-LoanDataset-Kaggle - Data Science Concept


## Summary
This project is designed to help practitioners practice various data science concepts, including:

- Training and tuning classification models
- Performing feature engineering to improve model performance
- Explaining, interpreting, and debugging models

## Tasks
This project involves the following tasks:

1. Load the dataset, excluding the "index" column for training.
2. Clean up the data by encoding and replacing missing values.
3. Split the dataset into Train/Test/Validation (or use cross-validation).
4. Add engineered features (at least 10) to enhance model performance.
5. Train and tune ML models.
6. Provide final metrics using the Test dataset, including AUC and confusion matrix.
7. Interpret the final trained model using various techniques, such as Shapley values and permutation feature importance.

## Datasets
The dataset provided originates from the U.S. Small Business Administration (SBA) and has been modified with dropped/added columns.

More information on the original dataset can be found [here](https://www.kaggle.com/mirbektoktogaraev/should-this-loan-be-approved-or-denied). However, for this project, use only the dataset provided with the project requirements.

## Deliverables
This project includes:

- `notebook` (.ipynb file)
  - Jupyter notebook with complete code (in both .ipynb)
- `artifacts` (folder)
  - Model and any encoders in "pkl" format or native H2O-3 format
  - Scoring function to load the final model and encoders
    

## Models
Train and tune two types of models: GBM (H2O) and LightGBM. 

- Split the dataset into Train/Validation/Test before applying any encodings or feature engineering.
- Encode categorical variables using suitable techniques.
- Perform hyperparameter tuning with at least 150 combinations or Optuna trials.
- Calculate the probability threshold to maximize F1.

## Scoring Function
single scoring function for either GBM or LightGBM. The scoring function:
- Accept a dataset in the same format as provided with the project (excluding the "target" column).
- Load the trained model and any necessary encoders.
- Transform the dataset for scoring and return results, including labels and probabilities.

## Model Interpretation
Provided a detailed write-up on what features are important for model predictions. Included:
- Shapley summary graph
- Shapley feature interaction graphs
- Multiple examples of single records Shapley graphs with explanations.
- Discuss the strong and weak points of the model.

## Resources
- [GitHub Repository](https://github.com/slundberg/shap)

## Model Performance (in H2O Driverless AI)
- AUC on cross-validation dataset: 0.8515
- AUC on hold-out dataset (not provided): 0.855

## Saves All Artifacts
Saved all artifacts needed for the scoring function, including the trained model and encoders.

## Model Scoring
Tested the model scoring function by restarting the Kernel and running all cells from top to bottom.

