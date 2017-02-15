import numerapi
import model
import time
import os
import numpy as np
import pandas as pd
from threading import Thread

def make_dir_if_necessary(dataset_id):
    dir_for_extra_features = os.path.join('generated_features', dataset_id)
    if not os.path.exists(dir_for_extra_features):
        os.makedirs(dir_for_extra_features)
    return dir_for_extra_features

def predict(train, tourn, dir_for_extra_features):
    print('Training model...')
    predictions_filename = model.process_data(train, tourn, dir_for_extra_features)
    print('Uploading predictions...')
    status = napi.upload_prediction(predictions_filename)
    while status != 200:
        print('Error while uploading predictions. Status code ', status)
        time.sleep(10)
        status = napi.upload_prediction(predictions_filename)
    print('Upload successful.')

def run_tsne_indefinitely(all_raw_features, dir_for_extra_features):
    print('Running t-SNE indefinitely in a separate thread. Uploading predictions after every finish.')
    while True:
        model.tsne(all_raw_features, dir_for_extra_features)
        predict(dir_for_extra_features)

def spawn_tsne_processes(train, tourn, dir_for_extra_features):
    raw_train_features, raw_tourn_features, all_raw_features = model.preprocess_data(train, tourn)
    process_count = 6
    for i in range(0, process_count):
        thread = Thread(target=run_tsne_indefinitely, args=(all_raw_features, dir_for_extra_features))
        thread.start()

napi = numerapi.NumerAPI()
prev_dataset_id = '589b897b1aed925a126572cb'
while True:
    status_code, dataset_id, comp_id = napi.get_current_competition()
    if status_code != 200:
        print('Error retrieving current competition details, status code ', status_code)
        time.sleep(10)
        continue
    dir_for_extra_features = make_dir_if_necessary(dataset_id)
    if dataset_id == prev_dataset_id:
        time.sleep(10)
        continue

    print('New dataset is available! Downloading... ', dataset_id)
    status = napi.download_current_dataset(dest_path="dataset", unzip=True)
    if status != 200:
        print('Error downloading! ', status)
        time.sleep(10)
        continue
    print('Download successful.')
    prev_dataset_id = dataset_id

    train = pd.read_csv('dataset/numerai_training_data.csv')
    tourn = pd.read_csv('dataset/numerai_tournament_data.csv')
    predict(train, tourn, dir_for_extra_features)
    spawn_tsne_processes(train, tourn, dir_for_extra_features)
    break
