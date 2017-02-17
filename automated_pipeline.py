import numerapi
import model
import time
import os
import numpy as np
import pandas as pd
from threading import Thread

def predict(raw_csv_train, raw_csv_tourn, dir_for_extra_features):
    predictions_filename = model.process_data(raw_csv_train, raw_csv_tourn, dir_for_extra_features)
    status = napi.upload_prediction(predictions_filename)
    while status != 200:
        print('Error while uploading predictions. Status code ', status)
        time.sleep(10)
        status = napi.upload_prediction(predictions_filename)
    print('Upload successful.')

def run_tsne_indefinitely(raw_csv_train, raw_csv_tourn, all_features, dir_for_extra_features):
    while True:
        model.tsne(all_features, dir_for_extra_features)
        predict(raw_csv_train, raw_csv_tourn, dir_for_extra_features)

def spawn_tsne_processes(raw_csv_train, raw_csv_tourn, dir_for_extra_features):
    train_features, tourn_features, all_features = model.preprocess_data(raw_csv_train, raw_csv_tourn)
    process_count = 5
    print('Running t-SNE indefinitely in', process_count, 'separate threads. Uploading predictions after every finish.')
    for i in range(0, process_count):
        thread = Thread(target=run_tsne_indefinitely, args=(raw_csv_train, raw_csv_tourn, all_features, dir_for_extra_features))
        thread.start()

napi = numerapi.NumerAPI()
#prev_dataset_id = os.path.basename(os.path.normpath(model.get_latest_generated_features_dir()))
prev_dataset_id = -1
while True:
    status_code, dataset_id, comp_id = napi.get_current_competition()
    if status_code != 200:
        print('Error retrieving current competition details, status code ', status_code)
        time.sleep(10)
        continue
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

    dir_for_extra_features = model.make_dir_if_necessary(dataset_id)
    raw_csv_train = pd.read_csv('dataset/numerai_datasets/numerai_training_data.csv')
    raw_csv_tourn = pd.read_csv('dataset/numerai_datasets/numerai_tournament_data.csv')
    predict(raw_csv_train, raw_csv_tourn, dir_for_extra_features)
    spawn_tsne_processes(raw_csv_train, raw_csv_tourn, dir_for_extra_features)
    break