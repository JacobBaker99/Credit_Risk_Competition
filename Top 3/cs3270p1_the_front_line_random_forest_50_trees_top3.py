#!/usr/bin/env python3

'''
Random Forest for Project One

This module is used to conduct a random Forest on train_person1.csv, train_person2.csv,
and train_base.csv. This is the best testing random forest and part of the top 3.
'''

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import ensemble
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer

__author__ = 'Frontline'
__version__ = 'Spring 2024'
__pylint__ = '2.14.5'

SEED = 3270

if __name__ == '__main__':
    # Load training (development) and testing data
    train = pd.read_csv('dev.csv')
    test = pd.read_csv('test.csv')

    # Drop unnecessary columns from training and testing datasets
    train.drop("case_id", axis=1, inplace=True)
    if "a-r" in test.columns:
        test.drop(["case_id", "a-r"], axis=1, inplace=True)


    # Convert data to string type
    train = train.astype(str)
    test = test.astype(str)

    # Concatenate train and test data for one-hot encoding
    combined_data = pd.concat([train, test])

    # Extract features and labels from training data
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

    # Perform one-hot encoding and handle categorical features
    one_hot = OneHotEncoder()
    trans = ColumnTransformer([("one_hot", one_hot, cat_features)], remainder="passthrough")
    encoded_features = trans.fit_transform(features).toarray()
    one_hot_features = trans.named_transformers_['one_hot'].get_feature_names_out(cat_features)
    remaining_features = [f for f in features.columns if f not in cat_features]
    all_feature_names = list(one_hot_features) + remaining_features

    # Split the combined data back into train and test
    encoded_features_train = encoded_features[:len(train)]
    encoded_features_test = encoded_features[len(train):]

    # Extract labels from training data
    labels_train = train['target']

    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(encoded_features_train)
    X_test_scaled = scaler.transform(encoded_features_test)

    # Define model parameters
    NUM_FOLDS = 5
    TREES = 50

    # Perform stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    for train_index, test_index in skf.split(X_train_scaled, labels_train):
        X_train, X_val = X_train_scaled[train_index], X_train_scaled[test_index]
        y_train, y_val = labels_train.iloc[train_index], labels_train.iloc[test_index]

        # Train random forest model
        rf_model = ensemble.RandomForestClassifier(criterion='entropy',
            n_estimators=TREES, random_state=SEED)
        rf_model.fit(X_train, y_train)
        print(f'\tRF_accuracy: {accuracy_score(y_val, rf_model.predict(X_val)) * 100:.2f}%')

    # Evaluate model performance with cross-validation
    rf_scores = cross_val_score(ensemble.RandomForestClassifier(criterion='entropy',
        n_estimators=TREES, random_state=SEED), X_train_scaled, labels_train,
        cv=skf, scoring="accuracy")
    print(f'Random Forest Scores: {rf_scores.mean():.2f} ± {rf_scores.std():.2f}')

    # Train final model and predict on testing data
    rf_model.fit(X_train_scaled, labels_train)
    predictions_test = rf_model.predict(X_test_scaled)

    # Evaluate predictions on testing data
    test_labels = test['target']
    print(f'\n\tRF_test_score: {accuracy_score(test_labels, predictions_test) * 100:.2f}%')
