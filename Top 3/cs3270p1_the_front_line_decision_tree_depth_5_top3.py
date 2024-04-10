#!/usr/bin/env python3

'''
Decision Tree for Project One

This module is used to conduct decision trees on train_person1.csv, train_person2.csv,
and train_base.csv. This is the decision tree with a max depth of 5.
'''

import warnings
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore")

__author__ = 'Frontline'
__version__ = 'Spring 2024'
__pylint__ = '2.14.5'

SEED = 3270

if __name__ == '__main__':
    data = pd.read_csv(r'dev.csv')
    data.drop("case_id", axis=1, inplace=True)
    data = data.astype(str)

    test = pd.read_csv(r'test.csv')
    test.drop("case_id", axis=1, inplace=True)
    test = test.astype(str)
    # Drop unnecessary columns from training and testing datasets
    if "a-r" in test.columns:
        test.drop(["case_id", "a-r"], axis=1, inplace=True)

    test_labels = test['target']

    for column in data.columns:
        if data[column].dtype == 'object':
            data[column].fillna("0", inplace=True)

    feature_list = [
        "birth_259D", "contaddr_smempladdr_334L", "empl_employedtotal_800L", "empl_industry_691L",
        "familystate_447L", "housetype_905L", "incometype_1044T", "mainoccupationinc_384A",
        "num_group1", "personindex_1023L", "persontype_1072L", "persontype_792L", "role_1084L",
        "safeguarantyflag_411L", "sex_738L", "type_25L"
    ]

    features = data[feature_list]
    labels = data['target']

    # One-hot encode categorical features
    cat_features = [
        "birth_259D", "contaddr_smempladdr_334L", "empl_employedtotal_800L", "empl_industry_691L",
        "familystate_447L", "housetype_905L", "incometype_1044T", "mainoccupationinc_384A",
        "num_group1", "personindex_1023L", "persontype_1072L", "persontype_792L", "role_1084L",
        "safeguarantyflag_411L", "sex_738L", "type_25L"
    ]

    one_hot = OneHotEncoder(handle_unknown='ignore')
    trans = ColumnTransformer([("one_hot", one_hot,
        cat_features)], remainder="passthrough")
    encoded_features = trans.fit_transform(features).toarray()
    one_hot_features = trans.named_transformers_['one_hot'].get_feature_names_out(cat_features)
    remaining_features = [f for f in features.columns if f not in cat_features]
    all_feature_names = list(one_hot_features) + remaining_features

    # Scale features
    X = pd.DataFrame(encoded_features, columns=all_feature_names)
    y = labels
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    NUM_FOLDS = 5
    DEPTH = 5

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    for train_index, test_index in skf.split(X_scaled, y):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        dt_model = DecisionTreeClassifier(criterion='entropy', random_state=SEED, max_depth=DEPTH)
        dt_model.fit(X_train, y_train)
        print(f'\tDT_entropy: {accuracy_score(y_test, dt_model.predict(X_test)) * 100:.2f}%')

    # Evaluate model performance with cross-validation
    dt_scores = cross_val_score(
        DecisionTreeClassifier(criterion='entropy', random_state=SEED, max_depth=DEPTH),
        X_scaled, y, cv=skf, scoring="accuracy"
    )
    print(f'Decision Tree Scores: {dt_scores.mean():.2f} ± {dt_scores.std():.2f}')

    # Train final model and predict on testing data
    encoded_test_features = trans.transform(test[feature_list]).toarray()
    X_test = pd.DataFrame(encoded_test_features, columns=all_feature_names)
    print(f'\n\tDecision Tree: {accuracy_score(test_labels, dt_model.predict(X_test)) * 100:.2f}%')
