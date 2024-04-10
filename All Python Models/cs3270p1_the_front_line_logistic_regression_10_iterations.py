#!/usr/bin/env python3

'''
Logistic Regression for Project One
10 iterations
'''

import pandas as pd
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer


# pylint: disable=C0103

__author__ = 'Frontline'
__version__ = 'Spring 2024'
__pylint__ = '2.14.5'

def prepData():
    '''
    Prepare the data for k-fold and log regression
    '''
    data = pd.read_csv(r'dev.csv')
    data.drop("case_id", axis=1, inplace=True)
    data = data.astype(str)

    for column in data.columns:
        if data[column].dtype == 'object':  # Checking for string/object type
            data[column].fillna("0", inplace=True)


    feature_list = [
        "birth_259D","contaddr_smempladdr_334L",
        "empl_employedtotal_800L","empl_industry_691L",
        "familystate_447L","housetype_905L","incometype_1044T",
        "mainoccupationinc_384A","num_group1","personindex_1023L",
        "persontype_1072L","persontype_792L","role_1084L",
        "safeguarantyflag_411L","sex_738L","type_25L"
    ]

    features = data[feature_list]

    labels = data['target']

    categorical_features = [
        "birth_259D","contaddr_smempladdr_334L","empl_employedtotal_800L",
        "empl_industry_691L","familystate_447L","housetype_905L",
        "incometype_1044T","mainoccupationinc_384A","num_group1",
        "personindex_1023L","persontype_1072L","persontype_792L",
        "role_1084L","safeguarantyflag_411L","sex_738L","type_25L"
    ]

    one_hot = OneHotEncoder()
    transformer = ColumnTransformer([("one_hot", one_hot,
                                       categorical_features)], remainder="passthrough")
    encoded_features = transformer.fit_transform(features).toarray()
    one_hot_features = transformer.named_transformers_['one_hot']
    one_hot_features = one_hot_features.get_feature_names_out(categorical_features)
    remaining_features = [f for f in features.columns if f not in categorical_features]
    all_feature_names = list(one_hot_features) + remaining_features
    assert len(all_feature_names) == encoded_features.shape[1], "Mismatch in feature count"
    return [encoded_features, all_feature_names, labels]



def analyze(encoded_features, all_feature_names, labels):
    '''
    Execute k fold and do the logistic regression 
    '''
    hyper = [3270, 5, 10]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(pd.DataFrame(encoded_features,
                        columns=all_feature_names))

    skf = StratifiedKFold(n_splits=hyper[1], shuffle=True, random_state=hyper[0])
    for train_index, test_index in skf.split(X_scaled, labels):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

        log_reg_model = linear_model.LogisticRegression(random_state=hyper[0],
                                                        fit_intercept=False, max_iter=10)
        log_reg_model.fit(X_train, y_train)
        print(f'\n\tLogReg: {accuracy_score(y_test, log_reg_model.predict(X_test)) * 100:.2f}%')



    # Repeat cross-validation with the scaled data
    lr_scores = cross_val_score(linear_model.LogisticRegression(max_iter=10),
                                X_scaled, labels, cv=skf, scoring="accuracy")
    print(f'\nLogistic Regression Scores: {lr_scores.mean():.2f} ± {lr_scores.std():.2f}')

if __name__ == "__main__":
    dataArray = prepData()
    analyze(dataArray[0], dataArray[1], dataArray[2])
