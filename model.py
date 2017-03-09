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
from sklearn import model_selection
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import log_loss
import os
from random import randint
pd.options.display.max_columns = 100

def make_dir_if_necessary(dataset_id):
    dir_for_extra_features = os.path.join('generated_features', dataset_id)
    if not os.path.exists(dir_for_extra_features):
        os.makedirs(dir_for_extra_features)
    return dir_for_extra_features

def filename_available(name):
    return not os.path.isfile(name)

def normalize(df):
    return (df - df.mean()) / (df.max() - df.min())

def loss(alg, features, targets, folds):
    # Training logloss
    predictions = alg.predict_proba(features)[:,1]
    train_logloss = log_loss(targets, predictions, normalize=True)
    # Cross validation logloss
    scores = model_selection.cross_val_score(alg, features, targets, cv=folds, scoring="neg_log_loss")
    cv_logloss = np.abs(scores.mean())
    # Combine both into a DataFrame
    d = [{'Logloss': train_logloss}, {'Logloss': cv_logloss}]
    return pd.DataFrame(d, index=["Training", "Cross val"])

# TODO: Aja L1 feature selection, sit L2 jäljelle jääneille featureille !

def write_predictions(alg, X_live, tourn):
    print('Creating predictions...')
    predictions = alg.predict_proba(X_live)[:, 1]

    # Reduce confidence
    #predictions = (predictions - 0.5) / 2.4 + 0.5

    # for i in range(0, len(predictions)):
    #     if predictions[i] > 0.5:
    #         predictions[i] = 0.5
    #     else:
    #         predictions[i] -= 0.000001

    submission = pd.DataFrame({
        '\"t_id\"': tourn["t_id"],
        '\"probability\"': predictions
    })
    # Write output to next available filename, like "output_predictions58.csv"
    out_filename = 'temp'
    for i in range(1, 1000000):
        out_filename = 'output_predictions' + str(i) + '.csv'
        out_filename = os.path.join('predictions', out_filename)
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
        out_filename = os.path.join(dir_for_extra_features, out_filename)
        if filename_available(out_filename):
            break
    bao_bhtsne.process_results(res, out_filename)

def collect_previously_extracted_features_from_files(dir):
    features = pd.DataFrame()
    for f in os.listdir(dir): # for each file or folder inside
        path = os.path.join(dir, f)
        if os.path.isfile(path) & path.endswith(".csv"): # read .csv files only
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
    feature_names = train.columns[0:train.columns.size-1] # exclude target from features
    raw_train_features = train[feature_names]
    raw_tourn_features = tourn[feature_names]
    all_raw_features = raw_train_features.append(raw_tourn_features)
    pca = sklearn.decomposition.PCA(n_components = 0.999, svd_solver='full')
    pca.fit(all_raw_features)
    all_features = pd.DataFrame(pca.transform(all_raw_features))
    train_features = pd.DataFrame(pca.transform(raw_train_features))
    tourn_features = pd.DataFrame(pca.transform(raw_tourn_features))
    return train_features, tourn_features, all_features

def process_data(train, tourn, dir_for_extra_features):
    train_features, tourn_features, all_features = preprocess_data(train, tourn)

    # Normalize features. Must be done for tourn and train set at the same time.
    norm_raw_features = normalize(all_features)
    train_features = norm_raw_features.iloc[:len(train_features), :]
    tourn_features = norm_raw_features.iloc[len(train_features):, :]

    extra_features = collect_previously_extracted_features_from_files(dir_for_extra_features)
    extra_train_features = extra_features[:len(train)]
    extra_tourn_features = extra_features[len(train):].reset_index(drop=True)
    tourn_features.reset_index(drop=True, inplace=True)
    eng_train_features = pd.concat([train_features, extra_train_features], axis=1)
    eng_tourn_features = pd.concat([tourn_features, extra_tourn_features], axis=1)

    print('Training model...')
    classifier_logistic_regression = LogisticRegression(max_iter=3000, tol=0.0000001, penalty='l1', C=0.03)
    classifier_logistic_regression.fit(eng_train_features, train['target'])
    #print(classifier_logistic_regression.coef_)
    print(loss(classifier_logistic_regression, eng_train_features, train["target"], 10))

    filename = write_predictions(classifier_logistic_regression, eng_tourn_features, tourn)
    return filename

def get_latest_generated_features_dir():
    b = 'generated_features'
    all_subdirs = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd): all_subdirs.append(bd)
    return max(all_subdirs, key=os.path.getmtime)

if __name__ == "__main__":
    dir_for_extra_features = get_latest_generated_features_dir()
    train = pd.read_csv('dataset/numerai_datasets/numerai_training_data.csv')
    tourn = pd.read_csv('dataset/numerai_datasets/numerai_tournament_data.csv')
    process_data(train, tourn, dir_for_extra_features)