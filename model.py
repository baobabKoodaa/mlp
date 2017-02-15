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
    return out_filename

def tsne(all_raw_features, dir_for_extra_features):
    import bao_bhtsne
    seed = randint(0, 10000)
    perplexity = randint(5, 50)
    out_dims = randint(2, 3)
    # Run BH-TSNE
    res = bao_bhtsne.run_bh_tsne(all_raw_features, verbose=True, randseed=seed, no_dims=out_dims, perplexity=perplexity, max_iter=2000)
    # Write output to next available filename, like "tsne_d3_p50_run4.csv"
    out_filename = 'temp'
    for i in range(1, 1000000):
        out_filename = 'tsne_out' + str(out_dims) + '_p' + str(perplexity) + '_run' + str(
            i) + '.csv'
        out_filename = join(dir_for_extra_features, out_filename)
        if filename_available(out_filename):
            break
    bao_bhtsne.process_results(res, out_filename)

def collect_previously_extracted_features_from_files(dir):
    features = pd.DataFrame()
    for f in listdir(dir): # for each file or folder inside
        path = join(dir, f)
        if isfile(path) & path.endswith(".csv"): # read .csv files only
            next_batch = pd.read_csv(path, header=None)

            # rename columns
            new_column_names = []
            for column in next_batch:
                modif = str(column) + str(f)
                new_column_names.append(modif)
            next_batch.columns = new_column_names

            features = pd.concat([features, next_batch], axis=1)
    return normalize(features)

def preprocess_data(train, tourn):
    feature_names = train.columns[0:50]
    raw_train_features = train[feature_names]
    raw_tourn_features = tourn[feature_names]
    all_raw_features = raw_train_features.append(raw_tourn_features)
    return raw_train_features, raw_tourn_features, all_raw_features

def process_data(train, tourn, dir_for_extra_features):
    raw_train_features, raw_tourn_features, all_raw_features = preprocess_data(train, tourn)

    # Normalize features. Must be done for tourn and train set at the same time.
    norm_raw_features = normalize(all_raw_features)
    raw_train_features = norm_raw_features.iloc[:len(raw_train_features), :]
    raw_tourn_features = norm_raw_features.iloc[len(raw_train_features):, :]

    extra_features = collect_previously_extracted_features_from_files(dir_for_extra_features)
    print(extra_features.columns)
    train_tsne_features = extra_features[:len(train)]
    tourn_tsne_features = extra_features[len(train):].reset_index(drop=True)
    eng_train_features = pd.concat([raw_train_features, train_tsne_features], axis=1)
    eng_tourn_features = pd.concat([raw_tourn_features, tourn_tsne_features], axis=1)

    print('Building model...')
    classifier_logistic_regression = LogisticRegression(max_iter=3000, tol=0.0000001, penalty='l1', C=0.03)
    classifier_logistic_regression.fit(eng_train_features, train['target'])
    #print(classifier_logistic_regression.coef_)
    print(loss(classifier_logistic_regression, eng_train_features, train["target"], 10))
    filename = write_predictions(classifier_logistic_regression, eng_tourn_features, tourn)
    return filename

if __name__ == "__main__":
    process_data()