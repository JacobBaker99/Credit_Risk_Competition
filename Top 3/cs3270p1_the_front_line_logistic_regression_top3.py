#!/usr/bin/env python3

'''
Logistic Regression for Project One

This module is used to conduct logistic Regression on train_person1.csv, train_person2.csv,
and train_base.csv. This is the logistic Regression that preformed the best with an iteration
of 100.
'''

import warnings
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.compose import ColumnTransformer

__author__ = 'Frontline'
__version__ = 'Spring 2024'
__pylint__ = '2.14.5'

SEED = 3270

if __name__ == '__main__':
    # Load training (development) and testing data
    dev_data = pd.read_csv('dev.csv')
    test_data = pd.read_csv('test.csv')

    # Drop unnecessary columns from training and testing datasets
    dev_data.drop("case_id", axis=1, inplace=True)
    if "a-r" in test_data.columns:
        test_data.drop(["case_id", "a-r"], axis=1, inplace=True)

    # Convert data to string type
    dev_data = dev_data.astype(str)
    test_data = test_data.astype(str)

    # Combine train and test data for consistent encoding
    combined_data = pd.concat([dev_data, test_data])

    # Extract features and labels from development data
    feature_list = [
        "birth_259D", "contaddr_smempladdr_334L", "empl_employedtotal_800L", "empl_industry_691L",
        "familystate_447L", "housetype_905L", "incometype_1044T", "mainoccupationinc_384A",
        "num_group1", "personindex_1023L", "persontype_1072L", "persontype_792L", "role_1084L",
        "safeguarantyflag_411L", "sex_738L", "type_25L"
    ]
    features = combined_data[feature_list]

    # Define categorical features for one-hot encoding
    cat_features = [
        "birth_259D", "contaddr_smempladdr_334L", "empl_employedtotal_800L",
        "empl_industry_691L", "familystate_447L", "housetype_905L", "incometype_1044T",
        "mainoccupationinc_384A", "num_group1", "personindex_1023L", "persontype_1072L",
        "persontype_792L", "role_1084L", "safeguarantyflag_411L", "sex_738L", "type_25L"
    ]

    # Perform one-hot encoding and handle unknown categories
    one_hot = OneHotEncoder(handle_unknown='ignore')
    trans = ColumnTransformer([("one_hot", one_hot, cat_features)], remainder="passthrough")
    encoded_features = trans.fit_transform(features)

    # Split the combined data back into train and test
    encoded_features_dev = encoded_features[:len(dev_data)]
    encoded_features_test = encoded_features[len(dev_data):]

    # Extract labels from development data
    dev_labels = dev_data['target']

    # Scale features
    scaler = MaxAbsScaler()
    X_dev_scaled = scaler.fit_transform(encoded_features_dev)

    # Define model parameters
    NUM_FOLDS = 5
    ITERATIONS = 5

    # Suppress warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Perform stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    for train_index, test_index in skf.split(X_dev_scaled, dev_labels):
        X_train, X_val = X_dev_scaled[train_index], X_dev_scaled[test_index]
        y_train, y_val = dev_labels.iloc[train_index], dev_labels.iloc[test_index]

        # Train logistic regression model
        log_reg_model = LogisticRegression(random_state=SEED,
            fit_intercept=False, max_iter=ITERATIONS)
        log_reg_model.fit(X_train, y_train)
        print(f'\taccuracy: {accuracy_score(y_val, log_reg_model.predict(X_val)) * 100:.2f}%')

    # Evaluate model performance with cross-validation
    log_reg_scores = cross_val_score(LogisticRegression(max_iter=ITERATIONS),
        X_dev_scaled, dev_labels,cv=skf, scoring="accuracy")
    print(f'Logistic Regression Scores: {log_reg_scores.mean():.2f} ± {log_reg_scores.std():.2f}')

    # Train final model and predict on testing data
    log_reg_model.fit(X_dev_scaled, dev_labels)
    X_test_scaled = scaler.transform(encoded_features_test)
    predictions_test = log_reg_model.predict(X_test_scaled)

    # Evaluate predictions on testing data
    test_labels = test_data['target']
    print(f'\n\tLogReg_test_score: {accuracy_score(test_labels, predictions_test) * 100:.2f}%')
