import datetime
import os

import flirt
import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

if os.uname()[1] == 'cluster_name':
    SOURCE_FOLDER = 'filepath'
    DATA_FOLDER_DATA = SOURCE_FOLDER + 'output/log-processing/'
    DATA_FOLDER_OUTPUT = SOURCE_FOLDER + 'output/features/'


def __merge_features(stat_features: pd.DataFrame, hrv_features: pd.DataFrame, freq: str = '1s') -> pd.DataFrame:
    if hrv_features.empty and stat_features.empty:
        print("Received empty input features, returning empty df")
        return pd.DataFrame()

    merged_features = pd.concat([hrv_features, stat_features], axis=1, sort=True)

    target_index = pd.date_range(start=merged_features.iloc[0].name.ceil('s'),
                                 end=merged_features.iloc[-1].name.floor('s'), freq=freq, tz=None)

    merged_features = merged_features.reindex(index=merged_features.index.union(target_index))

    merged_features.interpolate(method='time', limit = 1, inplace=True)

    merged_features = merged_features.reindex(target_index)

    return merged_features

#Use merge_asof instead
def __merge_raw_data(RESPIRATION: pd.DataFrame, HEART_RATE: pd.DataFrame, PULSE_OX: pd.DataFrame, ZERO_CROSSING: pd.DataFrame, STRESS: pd.DataFrame, freq: datetime.timedelta) -> pd.DataFrame:

    if RESPIRATION.empty and HEART_RATE.empty and PULSE_OX.empty and ZERO_CROSSING.empty and STRESS.empty:
        print("Received empty input features, returning empty df")
        return pd.DataFrame()

    interpolation_limit_dict = {'RESPIRATION': datetime.timedelta(minutes=5),
                                'HEART_RATE': datetime.timedelta(minutes=1),
                                'PULSE_OX': datetime.timedelta(minutes=5),
                                'ZERO_CROSSING': datetime.timedelta(minutes=1),
                                'STRESS': datetime.timedelta(minutes=5)}

    type_columns = ['RESPIRATION', 'HEART_RATE', 'PULSE_OX', 'ZERO_CROSSING', 'STRESS']
    merged_data = pd.concat([RESPIRATION, HEART_RATE, PULSE_OX, ZERO_CROSSING, STRESS], axis=1, sort=True)
    merged_data = merged_data.replace(-1, np.nan)
    merged_data = merged_data.replace(-2, np.nan)

    target_index = pd.date_range(start=merged_data.iloc[0].name.ceil('s'),
                                 end=merged_data.iloc[-1].name.floor('s'), freq=freq, tz=None)

    merged_data = merged_data.reindex(index=merged_data.index.union(target_index))
    for column in interpolation_limit_dict.keys():
        merged_data[column].interpolate(method='time', inplace=True, limit=interpolation_limit_dict[column]// freq)

    merged_data = merged_data.reindex(target_index)
    return merged_data


def feature_calc(DATA_FOLDER_DATA, WINDOW_LENGTH, WINDOW_STEP, INTERPOL_RAW, GET_STATISTICAL_FEATURES, GET_HRV_FEATURES):
    return_data = pd.DataFrame()

    for ID in ['radar-01', 'radar-10', 'radar-12', 'radar-13', 'radar-14', 'radar-15', 'radar-17', 'radar-18', 'radar-19',
                'radar-20', 'radar-21', 'radar-22', 'radar-23', 'radar-24', 'radar-25', 'radar-26', 'radar-27', 'radar-28',
                'radar-29', 'radar-30', 'radar-31', 'radar-32', 'radar-33', 'radar-34', 'radar-35', 'radar-36', 'radar-37',
                'radar-38', 'radar-39', 'radar-40']:

        print()
        print('processing %s' % ID)
        output_data = pd.DataFrame()

        file = DATA_FOLDER_DATA + "subjects/" + ID + ".parquet.gz"
        extract = pd.read_parquet(file)

        STRESS = pd.DataFrame(extract.loc[extract['type'] == 'STRESS']['value']).rename(columns={"value": "STRESS"})
        RESPIRATION = pd.DataFrame(extract.loc[extract['type'] == 'RESPIRATION']['value']).rename(columns={"value": "RESPIRATION"})
        HEART_RATE = pd.DataFrame(extract.loc[extract['type'] == 'HEART_RATE']['value']).rename(columns={"value": "HEART_RATE"})
        PULSE_OX = pd.DataFrame(extract.loc[extract['type'] == 'PULSE_OX']['value']).rename(columns={"value": "PULSE_OX"})
        ZERO_CROSSING = pd.DataFrame(extract.loc[extract['type'] == 'ZERO_CROSSING']['value']).rename(columns={"value": "ZERO_CROSSING"})

        STAT_DATA = pd.DataFrame(__merge_raw_data(RESPIRATION, HEART_RATE, PULSE_OX, ZERO_CROSSING, STRESS, freq = INTERPOL_RAW))

        HEART_RATE_VARIABILITY = pd.Series(extract.loc[extract['type'] == 'HEART_RATE_VARIABILITY']['value'])

        if GET_STATISTICAL_FEATURES:
            #try:
            stat_features = flirt.get_stat_features(STAT_DATA[['HEART_RATE', 'RESPIRATION', 'PULSE_OX', 'ZERO_CROSSING', 'STRESS']],
                                                        window_step_size=WINDOW_STEP,#//int(INTERPOL_RAW.total_seconds()),
                                                        window_length=WINDOW_LENGTH,
                                                        num_cores=6,
                                                        data_frequency=1)
            #except: pass
        if GET_HRV_FEATURES:
            #try:
            hrv_features = flirt.get_hrv_features(HEART_RATE_VARIABILITY,
                                                      domains=['td', 'fd', 'stat'],
                                                      window_step_size=WINDOW_STEP,
                                                      num_cores=6,
                                                      window_length=WINDOW_LENGTH)
            #except: pass

        if (GET_STATISTICAL_FEATURES == True) and (GET_HRV_FEATURES == True):
            print("Save set of HRV and Statistical Features of %s" % (ID))
            output_data = pd.merge_asof(stat_features, hrv_features,
                                            left_index=True, right_index=True,
                                            direction='nearest',
                                            tolerance=datetime.timedelta(seconds=WINDOW_LENGTH // 2))
            output_data['ID'] = ID
            output_data.to_parquet(DATA_FOLDER_OUTPUT + 'merged_features_' + ID + '_' + str(WINDOW_LENGTH) + '_'
                               + str(WINDOW_STEP) + '.parquet.gzip', compression='gzip')


        elif (GET_STATISTICAL_FEATURES == False) and (GET_HRV_FEATURES == True):
            print("Save set of HRV Features of %s" % (ID))
            output_data = hrv_features
            output_data['ID'] = ID
            output_data.to_parquet(DATA_FOLDER_OUTPUT + 'merged_features_' + ID + '_' + str(WINDOW_LENGTH) + '_'
                               + str(WINDOW_STEP) + '.parquet.gzip', compression='gzip')

        elif (GET_STATISTICAL_FEATURES == True) and (GET_HRV_FEATURES == False):
            print("Save set of Statistical Features of %s" % (ID))
            output_data = stat_features
            output_data['ID'] = ID
            output_data.to_parquet(DATA_FOLDER_OUTPUT + 'stat_features_' + ID + '_' + str(WINDOW_LENGTH) + '_'
                               + str(WINDOW_STEP) + '.parquet.gzip', compression='gzip')

    if (GET_STATISTICAL_FEATURES == False) and (GET_HRV_FEATURES == False):
        print("No feature task was given! Return empty DataFrame")
        return pd.DataFrame()

    else:
        return_data = return_data.append(output_data)
        return return_data


if __name__ == '__main__':
    WINDOW_LENGTH = int(60)
    WINDOW_STEPS = [int(60)]
    INTERPOL_RAW = datetime.timedelta(seconds = 1)
    GET_STATISTICAL_FEATURES = True
    GET_HRV_FEATURES = True

    for WINDOW_STEP in WINDOW_STEPS:
        print(f'WINDOW_STEP: {WINDOW_STEP}')
        _ = feature_calc(DATA_FOLDER_DATA = DATA_FOLDER_DATA,
                         WINDOW_LENGTH = WINDOW_LENGTH,
                         WINDOW_STEP = WINDOW_STEP,
                         INTERPOL_RAW = INTERPOL_RAW,
                         GET_STATISTICAL_FEATURES = GET_STATISTICAL_FEATURES,
                         GET_HRV_FEATURES = GET_HRV_FEATURES)



