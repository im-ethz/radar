import datetime
import os

import flirt
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob2 as glob
import flirt.simple
import flirt.eda.preprocessing as eda_preprocess

os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

if os.uname()[1] == 'cluster_name':
    SOURCE_FOLDER = 'filepath'
    DATA_FOLDER_DATA = SOURCE_FOLDER + 'output/log-processing/'
    DATA_FOLDER_OUTPUT = SOURCE_FOLDER + 'output/features/'

def feature_calc(IDS: list, WINDOW_LENGTH: int, WINDOW_STEP:int,RADAR_FILE_PATH: str):

    for id in IDS:

        df = pd.DataFrame()
        files = glob.glob(RADAR_FILE_PATH + '/' + id + '/*.zip')
        df_eda_features = pd.DataFrame()
        for file in files:

            with zipfile.ZipFile(file, 'r') as zip_file:
                with zip_file.open("EDA.csv") as f:
                    eda_data = flirt.reader.empatica.read_eda_file_into_df(f)

            n = 12 * 4 * 3600  # chunk row size
            list_df = [eda_data[i:i + n] for i in range(0, eda_data.shape[0], n)]

            for element in list_df:

                df_eda_features = df_eda_features.append(flirt.eda.get_eda_features(element,
                                                                                    window_length=WINDOW_LENGTH,
                                                                                    window_step_size=WINDOW_STEP,
                                                                                    preprocessor=eda_preprocess.LowPassFilter(),
                                                                                    signal_decomposition=eda_preprocess.CvxEda(),
                                                                                    scr_features=eda_preprocess.ComputeMITPeaks(),
                                                                                    num_cores=6))
        df_eda_features.sort_index(inplace=True)
        print("Save set of Features of %s" % (id))
        df_eda_features.to_parquet(DATA_FOLDER_OUTPUT + 'empatica_features_' + id + '_' + str(WINDOW_LENGTH) + '_'
                               + str(WINDOW_STEP) + '.parquet.gzip', compression='gzip')

if __name__ == '__main__':
    WINDOW_LENGTH = int(60)
    WINDOW_STEPS = [int(60)]

    RADAR_FILE_PATH = SOURCE_FOLDER + "empatica-entries"
    IDS = ['radar-01', 'radar-10', 'radar-12', 'radar-13', 'radar-14', 'radar-15', 'radar-17', 'radar-18', 'radar-19',
           'radar-20', 'radar-21', 'radar-22', 'radar-23', 'radar-24', 'radar-25', 'radar-26', 'radar-27', 'radar-28', 'radar-29',
           'radar-30', 'radar-31', 'radar-32', 'radar-33', 'radar-34', 'radar-35', 'radar-36', 'radar-37', 'radar-38', 'radar-39',
           'radar-40']


    for WINDOW_STEP in WINDOW_STEPS:
        print(f'WINDOW_STEP: {WINDOW_STEP}')
        feature_calc(IDS=IDS, WINDOW_LENGTH=WINDOW_LENGTH, WINDOW_STEP=WINDOW_STEP, RADAR_FILE_PATH=RADAR_FILE_PATH)



