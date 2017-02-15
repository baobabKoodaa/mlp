import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import sklearn
from sklearn.manifold import Isomap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn import cross_validation
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import log_loss
from os import listdir
from os.path import isfile, join
from random import randint
pd.options.display.max_columns = 100

def filename_available(name):
    return not isfile(name)

def normalize(df):
    return (df - df.mean()) / (df.max() - df.min())

def loss(alg, features, targets, folds):
    # Training logloss
    predictions = alg.predict_proba(features)[:,1]
    train_logloss = log_loss(targets, predictions, normalize=True)
    # Cross validation logloss
    scores = cross_validation.cross_val_score(alg, features, targets, cv=folds, scoring="log_loss")
    cv_logloss = np.abs(scores.mean())
    # Combine both into a DataFrame
    d = [{'Logloss': train_logloss}, {'Logloss': cv_logloss}]
    return pd.DataFrame(d, index=["Training", "Cross val"])

# TODO: Aja L1 feature selection, sit L2 jäljelle jääneille featureille !

def write_predictions(alg, X_live, tourn):
    predictions = alg.predict_proba(X_live)[:, 1]
    submission = pd.DataFrame({
        '\"t_id\"': tourn["t_id"],
        '\"probability\"': predictions
    })
    # Write output to next available filename, like "output_predictions58.csv"
    out_filename = 'temp'
    for i in range(1, 1000000):
        out_filename = 'output_predictions' + str(i) + '.csv'
        out_filename = join('predictions', out_filename)
        if filename_available(out_filename):
            break
    submission.to_csv(out_filename, columns=('\"t_id\"', '\"probability\"'), index=False,
                      quoting=csv.QUOTE_NONE)
    print('Done writing predictions to ', out_filename)

def train_model():
    train = pd.read_csv('dataset/numerai_training_data.csv')
    tourn = pd.read_csv('dataset/numerai_tournament_data.csv')
    feature_names = train.columns[0:50]
    raw_train_features = train[feature_names]
    raw_tourn_features = tourn[feature_names]
    print(feature_names)

    # Normalize features. Must be done for tourn and train set at the same time.
    all_raw_features = raw_train_features.append(raw_tourn_features)
    norm_raw_features = normalize(all_raw_features)
    raw_train_features = norm_raw_features.iloc[:len(raw_train_features), :]
    raw_tourn_features = norm_raw_features.iloc[len(raw_train_features):, :]

    print('Building model...')
    classifier_logistic_regression = LogisticRegression(max_iter=3000, tol=0.0000001, penalty='l1', C=0.03)
    classifier_logistic_regression.fit(raw_train_features, train['target'])
    print(classifier_logistic_regression.coef_)
    print(loss(classifier_logistic_regression, raw_train_features, train["target"], 10))
    write_predictions(classifier_logistic_regression, raw_tourn_features, tourn)


train_model()