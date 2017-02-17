# Modified for Python3, original Python2 version from: https://github.com/atreichel/NumerAPI/blob/master/numerapi.py
# -*- coding: utf-8 -*-
import requests
import zipfile
from datetime import datetime, timedelta
import numpy as np
import os
import ntpath
import time

class NumerAPI(object):
    def __init__(self):
        self._login_url = 'https://api.numer.ai/sessions'
        self._auth_url = 'https://api.numer.ai/upload/auth'
        self._dataset_url = 'https://api.numer.ai/competitions/current/dataset'
        self._submissions_url = 'https://api.numer.ai/submissions'
        self._users_url = 'https://api.numer.ai/users'
        # Login email and password are in environment variables
        self._payload = {'email':os.environ['NUMERAI_EMAIL'], 'password':os.environ['NUMERAI_PASSWORD']}

    def download_current_dataset(self, dest_path='.', unzip=True):
        now = datetime.now().strftime('%Y%m%d')
        file_name = 'numerai_dataset_{0}.zip'.format(now)
        dest_file_path ='{0}/{1}'.format(dest_path, file_name)

        r = requests.get(self._dataset_url, stream=True)
        if r.status_code!=200:
            return r.status_code

        with open(dest_file_path, "wb") as fp:
            for block in r.iter_content(1024):
                fp.write(block)

        if unzip:
            with zipfile.ZipFile(dest_file_path, "r") as z:
                z.extractall(dest_path)
            z.close()
        return r.status_code


    def get_leaderboard(self):
        now = datetime.now()
        tdelta = timedelta(microseconds=55296e5)
        dt = now - tdelta
        dt_str = dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

        url = 'https://api.numer.ai/competitions?{ leaderboard :'
        url += ' current , end_date :{ $gt : %s }}'
        r = requests.get((url % (dt_str)).replace(' ', '%22'))
        if r.status_code!=200:
            return (None, r.status_code)
        return (r.json(), r.status_code)



    def get_earnings_per_round(self, username):
        r = requests.get('{0}/{1}'.format(self._users_url, username))
        if r.status_code!=200:
            return (None, r.status_code)

        rj = r.json()
        rewards = rj['rewards']
        earnings = np.zeros(len(rewards))
        for i in range(len(rewards)):
            earnings[i] = rewards[i]['amount']
        return (earnings, r.status_code)



    def get_scores(self, username):
        r = requests.get('{0}/{1}'.format(self._users_url, username))
        if r.status_code!=200:
            return (None, r.status_code)

        rj = r.json()
        results = rj['submissions']['results']
        scores = np.zeros(len(results))
        for i in range(len(results)):
            scores[i] = results[i]['accuracy_score']
        return (scores, r.status_code)



    def get_user(self, username):
        leaderboard, status_code = self.get_leaderboard()
        if status_code!=200:
            return (None, None, None, None, status_code)

        for user in leaderboard[0]['leaderboard']:
            if user['username']==username:
                return (user['username'], np.float(user['logloss']['public']),  user['rank']['public'],  user['earned'], status_code)
        return (None, None, None, None, status_code)



    def login(self):
        r = requests.post(self._login_url, data=self._payload)
        if r.status_code!=201:
            return (None, None, None, r.status_code)

        rj = r.json()
        return(rj['accessToken'], rj['refreshToken'], rj['id'], r.status_code)



    def authorize(self, file_path):
        accessToken, refreshToken, id_, status_code = self.login()
        if status_code!=201:
            return (None, None, None, status_code)

        headers = {'Authorization':'Bearer {0}'.format(accessToken)}

        filename = ntpath.split(file_path)[1]
        r = requests.post(self._auth_url,
                    data={'filename':filename, 'mimetype': 'text/csv'},
                    headers=headers)
        if r.status_code!=200:
            return (None, None, None, r.status_code)

        rj = r.json()
        return (rj['filename'], rj['signedRequest'], headers, r.status_code)

    def get_current_competition(self):
        while True:
            now = datetime.now()
            leaderboard, status_code = self.get_leaderboard()
            if status_code != 200:
                print('Error retrieving current competition details, status code ', status_code)
                time.sleep(10)
                continue

            for c in leaderboard:
                start_date = datetime.strptime(c['start_date'], '%Y-%m-%dT%H:%M:%S.%fZ')
                end_date = datetime.strptime(c['end_date'], '%Y-%m-%dT%H:%M:%S.%fZ')
                if start_date < now < end_date:
                    return (status_code, c['dataset_id'], c['_id'])



    def upload_prediction(self, file_path):
        while True:
            print('Uploading predictions...')
            filename, signedRequest, headers, status_code = self.authorize(file_path)
            if status_code!=200:
                print('Authorization error!', r.status_code)
                return

            status_code, dataset_id, comp_id = self.get_current_competition()
            with open(file_path, 'rb') as fp:
                r = requests.Request('PUT', signedRequest, data=fp.read())
                prepped = r.prepare()
                s = requests.Session()
                resp = s.send(prepped)
                if resp.status_code!=200:
                    return resp.status_code

            r = requests.post(self._submissions_url,
                        data={'competition_id':comp_id, 'dataset_id':dataset_id, 'filename':filename},
                        headers=headers)
            if r.status_code == 200:
                print('Upload successful.')
                return
            print('Error while uploading predictions. Status code ', r.status_code)
            time.sleep(10)

'''
MIT License

Copyright (c) 2016

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.MIT License

Copyright (c) 2016

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''