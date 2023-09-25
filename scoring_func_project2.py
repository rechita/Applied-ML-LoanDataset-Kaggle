#!/usr/bin/env python
# coding: utf-8

# In[4]:


def project_2_scoring(data):
    """
    Function to score input dataset.
    
    Input: dataset in Pandas DataFrame format
    Output: Python list of labels in the same order as input records
    
    Flow:
        - Load artifacts
        - Transform dataset
        - Score dataset
        - Return labels
    
    """
    from copy import deepcopy
    import pickle
    import pandas as pd
    import numpy as np
    from category_encoders import TargetEncoder
    from sklearn import metrics
    from sklearn.metrics import roc_curve, f1_score, precision_recall_curve
    
     
    X = data.copy()
    
    '''Load Artifacts'''
    artifacts_dict_file = open("./artifacts/artifacts_dict_file.pkl", "rb")
    artifacts_dict = pickle.load(file=artifacts_dict_file)
    artifacts_dict_file.close()
    
    clf = artifacts_dict["model"]
    target = artifacts_dict["target_column"]
    encoders = artifacts_dict["cat_encoders"]
    categorical_columns = artifacts_dict["categorical_columns"]
    predictors =  artifacts_dict["predictors"]
    threshold = artifacts_dict["threshold"]
            
    
    # Missing value treatment
    values_to_fill = {}
    for col in X.drop(columns=['MIS_Status']).columns:
        if X[col].dtype == 'object':
            values_to_fill[col] = "Missing"
        else:
            values_to_fill[col] = 0

    X.fillna(value=values_to_fill,inplace=True)

    
    # Data Cleaning process
    cols_to_convert = ['DisbursementGross', 'BalanceGross', 'GrAppv', 'SBA_Appv']
    X[cols_to_convert] = X[cols_to_convert].replace('[\$,]', '', regex=True).astype(float)
    
    
    # Creating New Features
    #ratio represents the percentage of the total loan amount that is guaranteed by the SBA
    X['ApprovalRatio'] = np.where(X['GrAppv'] == 0, 0, X['SBA_Appv'] / X['GrAppv'])
    # the average number of employees per job created
    X['EmpPerCreateJob'] = np.where(X['CreateJob'] == 0, 0, X['NoEmp'] / X['CreateJob'])
    # average number of employees per job retained
    X['EmpPerRetainedJob'] = np.where(X['RetainedJob'] == 0, 0, X['NoEmp'] / X['RetainedJob'])
    # percentage of the approved loan amount that was actually disbursed
    X['DisbursementRatio'] = np.where(X['GrAppv'] == 0, 0, X['DisbursementGross'] / X['GrAppv'])
    # percentage of the disbursed amount that is guaranteed by the SBA
    X['SBA_AppvRatio'] = np.where(X['DisbursementGross'] == 0, 0, X['SBA_Appv'] / X['DisbursementGross'])
    # Total number of Jobs for businesses
    X['TotalJobs'] = X['CreateJob'] + X['RetainedJob']
    # ratio of the disbursement amount to the amount approved by the SBA
    X['Disbursement_to_Appv_ratio'] = np.where(X['SBA_Appv'] == 0, 0, X['DisbursementGross'] / X['SBA_Appv'])
    # measure of how much of the loan has been paid off so far
    X['NetDisbursement'] = np.where(X['BalanceGross'] == 0, 0, X['DisbursementGross'] / X['BalanceGross'])
    # feature indicating the average amount of disbursement gross per employee
    X['DisbursementGross_per_employee'] = np.where(X['NoEmp'] == 0, 0, X['DisbursementGross'] / X['NoEmp'])
    # feature indicating the average amount of net disbursement per employee
    X['DisbursementNet_per_employee'] = np.where(X['NoEmp'] == 0, 0, X['NetDisbursement'] / X['NoEmp'])


    
    # Encoded categorical columns
    for col in categorical_columns:
        target_encoder = encoders[col][0]
        target_encoder.fit(X[col], X[target])
        X = X.join(target_encoder.transform(X[col]).add_suffix("_tr"))
    
    
    # Converting variables in Modified Dataset    
    for column in categorical_columns:
        X[column] = X[column].astype('category')
    
    # Get the predicted probabilities for the validation set
    y_proba = clf.predict_proba(X[predictors])
    y_proba_val = clf.predict_proba(X[predictors])[:, 1]
    y_pred_val = (y_proba_val > threshold).astype(int)
    # Generate the confusion matrix
    d = {"index":X["index"],
         "label":y_pred_val,
         "probability_0":y_proba[:,0],
         "probability_1":y_proba[:,1]} 
    
    return pd.DataFrame(d) 
    


# In[ ]:





# In[ ]:





# In[ ]:




