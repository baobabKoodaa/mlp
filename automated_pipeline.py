import numerapi
import model
import time

napi = numerapi.NumerAPI()
prev_dataset_id = '-1' # ''589b897b1aed925a126572cb'
while True:
    status_code, dataset_id, comp_id = napi.get_current_competition()
    if status_code != 200:
        print('Error retrieving current competition details, status code ', status_code)
        time.sleep(10)
        continue
    if dataset_id == prev_dataset_id:
        # Wait for new dataset
        time.sleep(10)
        continue
    print('New dataset is available! Downloading... ', dataset_id)
    status = napi.download_current_dataset(dest_path="dataset", unzip=True)
    if status != 200:
        print('Error downloading! ', status)
        continue
    print('Download successful. Training model...')
    predictions_filename = model.process_data()
    print('Uploading predictions...')
    status = napi.upload_prediction(predictions_filename)
    while status != 200:
        print('Error while uploading predictions. Status code ', status)
        time.sleep(10)
        status = napi.upload_prediction(predictions_filename)
    print('Upload successful.')
    prev_dataset_id = dataset_id