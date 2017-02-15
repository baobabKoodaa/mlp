import numerapi
import model
import time

napi = numerapi.NumerAPI()
prev_comp_id = -1
while True:
    curr_comp = napi.get_current_competition()
    if curr_comp[0] != 200:
        print('Error retrieving current competition details, status code ', curr_comp[0])
        time.sleep(10)
        continue
    if curr_comp[0] == prev_comp_id:
        # Wait for new dataset
        time.sleep(10)
        continue
    print('New dataset is available! Downloading... ', curr_comp[0])
    status = napi.download_current_dataset(dest_path="dataset", unzip=True)
    if status != 200:
        print('Error downloading! ', status)
        continue
    print('Download successful. Training model...')
    model.train_model()


    prev_comp_id = curr_comp[0]