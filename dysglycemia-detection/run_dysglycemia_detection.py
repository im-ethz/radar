import os
import sys
import hashlib
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Find code directory relative to our directory
abspath = os.path.abspath(__file__)
os.chdir(os.path.dirname(abspath))

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../../../../..')))
THIS_FILE = os.path.abspath(__file__)
#os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

import datetime as dt
import shap
import itertools

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product
from joblib import Parallel, delayed
from sklearn import model_selection, metrics
from sklearn import preprocessing
from pathlib import Path

from helper import remove_highly_correlated_features, reader, plot_glucose_distribution

if os.uname()[1] == 'mtec-im-gpu01':
    ROOT = '/wave/radar/'
    RESULTS_DIR = ROOT + 'output/results/'
else:
    ROOT = './data'
    RESULTS_DIR = './results/'

FEATURE_DIR = ROOT + 'output/features/'
BG_DIR = ROOT + 'dexcom-entries/'

def create_filepath():
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = parent_dir + '/data/' + ('hypo' if GLUCOSE_THRESHOLD == 3.9 else 'hyper')
    if (TIME == True) & (TIME != "only"):
        str_time = "time"
    elif TIME == False:
        str_time = "no_time"
    elif TIME == "only":
        str_time = "time_only"

    results_dir = parent_dir + '/results/' + f'{DATE}/' \
                                             f'{PID}/' \
                                             f'{"hypo" if GLUCOSE_THRESHOLD == 3.9 else "hyper"}/' \
                                             f'{SEED}/' \
                                             f'WL{WINDOW_LENGTH}_WS{WINDOW_STEP}/' \
                                             f'{"calibration" if CALIBRATION else "wocalibration"}/' \
                                             f'{str_time}/'
                                             #f'_{"extended" if EXTEND_TRAINING_DATA else "not_extended"}' \

    parquet_filename = data_dir + 'all_data_%s.parquet'
    numpy_filename = results_dir + '%s.npy'

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    return data_dir, results_dir, parquet_filename, numpy_filename

def scale_features(df):
    features = df.drop(columns=['label', 'ID', 'CGM value [mmol/L]']).columns
    df[features] = preprocessing.RobustScaler().fit_transform(df[features])
    return df


def evaluate(model, X_test, y_test, X_train, y_train, X_val = None, y_val = None, threshold=0.5, setting = None,
             ID = None, split_end = None, split_begin = None):

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)  # y_pred = np.argmax(y_pred_proba, axis=1)

    if not X_val is None:
        y_pred_proba_val = model.predict_proba(X_val)[:, 1]
        y_pred_val = (y_pred_proba_val >= threshold).astype(int)

    score = {}

    TN, FP, FN, TP = metrics.confusion_matrix(y_test, y_pred).ravel()

    score['Specificity'] = TN / (TN + FP)
    score['Sensitivity'] = TP / (TP + FN)
    score['F1 Score'] = metrics.f1_score(y_test, y_pred)
    score['MCC'] = metrics.matthews_corrcoef(y_test, y_pred)
    score['BACC'] = metrics.balanced_accuracy_score(y_test, y_pred)
    score['AUROC'] = metrics.roc_auc_score(y_test, y_pred_proba)
    score['AUROC_val'] = metrics.roc_auc_score(y_val, y_pred_proba_val) if not X_val is None else np.nan
    score['AUPRC'] = metrics.average_precision_score(y_test, y_pred_proba)

    x = np.linspace(0, 1, 100)
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    tpr = np.interp(x, fpr, tpr)
    tpr[0] = 0.0

    precision, recall, _ = metrics.precision_recall_curve(y_test, y_pred_proba)
    precision = np.flip(np.interp(x, np.flip(recall), np.flip(precision)))
    precision[-1] = 1.0

    if SHAP:

        explainer = shap.TreeExplainer(model)
        X_shap, _ , _, _ = model_selection.train_test_split(X_train, y_train, train_size=0.05)
        explainer_values = explainer(X_shap)
        importances = pd.DataFrame(index=explainer_values.feature_names, data={'importance': explainer_values[:, :, 1].abs.mean(0).values}).sort_values(by='importance', ascending=False)
        hash_hex = hashlib.sha256((str(split_end)+str(split_begin)).encode()).hexdigest()
        importance_dir = results_dir + f"/sh_imp_{setting}_{ID}_{hash_hex}.csv"
        importances.to_csv(importance_dir)

        #SAVE_AW_SHAP = False # Warning: Requires an insane amount of storage.
        #if SAVE_AW_SHAP:
        #    shap_dir = results_dir + f"/sh_raw_{setting}_{ID}_{hash_hex}.pkl"
        #    with open(shap_dir, "wb") as f:
         #       pickle.dump(explainer_values, f)

    return score, tpr, precision


def personalized(data):

    ids_personalized = np.unique([x for x in list(data_hypo['ID']) if list(data_hypo['ID']).count(x) > 1])

    print('\n', '=' * 80)
    print(f'Begin training personalied model (n_test={len(ids_personalized)}) ...')
    print(f'\n Test participants: \n\n', ids_personalized)
    print(f'\n Model parameters: \n\n', GBDT_PARAMS)
    feat = data.drop(columns=['label', 'ID', 'CGM value [mmol/L]']).columns
    print(f'\n Features (no. {len(feat)}):', *feat, sep=f'\n')

    print('\n','=' * 80)

    score, tpr, precision, shap_values, shap_base, shap_data = {}, {}, {}, {}, {}, {}
    results = pd.DataFrame(columns=['AUROC', 'AUPRC', 'BACC', 'F1 Score', 'MCC', 'Sensitivity', 'Specificity'])

    with Parallel(n_jobs=16, verbose=0) as parallel:

        for ID in ids_personalized:

            dataset = data.copy()
            data_run = dataset.loc[(dataset.ID == ID)].copy()
            X = data_run.drop(columns=['label', 'ID', 'CGM value [mmol/L]'])
            y = data_run['label'].astype('int')

            skf = model_selection.StratifiedKFold(n_splits=K)
            skf.get_n_splits(X, y)

            score[ID] = results.copy()
            tpr[ID], precision[ID] = pd.DataFrame(columns=np.arange(100)), pd.DataFrame(columns=np.arange(100))

            def run_personalized(train_index, test_index):

                classifier = lgb.LGBMClassifier(**GBDT_PARAMS)

                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                transformer = preprocessing.RobustScaler().fit(X_train)
                X_train, X_test = transformer.transform(X_train), transformer.transform(X_test)
                if len(X_test) > 0 and np.count_nonzero(y_test) > 0 and len(y_test[y_test == 0]) > 0:

                    classifier.fit(pd.DataFrame(X_train, columns=transformer.feature_names_in_), y_train)

                    return evaluate(classifier,
                                    pd.DataFrame(X_test, columns=transformer.feature_names_in_), y_test,
                                    pd.DataFrame(X_train, columns=transformer.feature_names_in_), y_train)

            result = parallel(delayed(run_personalized)(train_index, test_index) for train_index, test_index in skf.split(X, y))

            score[ID] = pd.DataFrame([i[0] for i in result if i is not None], columns=results.columns)
            tpr[ID] = pd.DataFrame([i[1] for i in result if i is not None])
            precision[ID] = pd.DataFrame([i[2] for i in result if i is not None])
            results.at[ID] = score[ID].mean()

            print('=' * 80)
            print(f'Results for {ID}')
            print(score[ID].agg(lambda x: f'{x.mean():.2f}±{x.std(ddof=0):.2f}'))
            print('=' * 80)
            print(f'Current overall result')
            print(results.agg(lambda x: f'{x.mean():.2f}±{x.std(ddof=0):.2f}'))
            print('=' * 80)

    return score, tpr, precision

def generalized(data):

    ids_generalized = np.unique([x for x in list(data_hypo['ID']) if list(data_hypo['ID']).count(x) >= 1])

    print('\n', '=' * 80)
    print(f'Begin training generalized model (n_test={len(ids_generalized)}) ...')
    print(f'\n Test participants: \n\n', ids_generalized)

    print(f'\n Model parameters: \n\n', GBDT_PARAMS)

    feat = data.drop(columns=['label', 'ID', 'CGM value [mmol/L]']).columns
    print(f'\n Features (no. {len(feat)}):', *feat, sep=f'\n')

    print('\n','=' * 80)

    dataset = data.copy()
    dataset = dataset.loc[dataset['ID'].isin(ids_generalized)]

    # Preliminaries summary
    score, tpr, precision, shap_values, shap_base, shap_data = {}, {}, {}, {}, {}, {}
    results = pd.DataFrame(columns=['AUROC', 'AUPRC', 'BACC', 'F1 Score', 'MCC', 'Sensitivity', 'Specificity'])
    # ============================

    # Run classifier with cross-validation
    cv = model_selection.LeaveOneGroupOut()


    X = dataset.drop(columns=['label', 'ID', 'CGM value [mmol/L]'])
    y = dataset['label'].astype('int')
    groups = dataset['ID']

    for train_index, test_index in cv.split(X, y, groups):
        ID = groups[test_index[0]]
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        classifier = lgb.LGBMClassifier(**GBDT_PARAMS)
        classifier.fit(X_train, y_train)

        result = evaluate(classifier, X_test, y_test, X_train, y_train)
        score[ID] = pd.DataFrame(np.array([result[0][k] for k in result[0].keys()]).reshape(1,8), columns=result[0].keys())
        tpr[ID] = pd.DataFrame(result[1])
        precision[ID] = pd.DataFrame(result[2])
        results.at[ID] = score[ID].mean()

        print('=' * 80)
        print(f'Results for {ID}')
        print(score[ID].agg(lambda x: f'{x.mean():.2f}±{x.std(ddof=0):.2f}'))
        print('=' * 80)
        print(f'Current overall result')
        print(results.agg(lambda x: f'{x.mean():.2f}±{x.std(ddof=0):.2f}'))
        print('=' * 80)

    return score, tpr, precision


def loho_gen(data, data_hypo):

    X = data.drop(columns = ['label', 'ID', 'CGM value [mmol/L]'])
    y = data['label'].astype('int')

    score, tpr, precision, shap_values, shap_base, shap_data = {}, {}, {}, {}, {}, {}
    results = pd.DataFrame(columns=['AUROC', 'AUROC_val', 'AUPRC', 'BACC', 'F1 Score', 'MCC', 'Sensitivity',
                                    'Specificity', 'ID_train', 'ID_test'])

    print('\n', '=' * 80)
    ids_test = pd.DataFrame(np.unique([x for x in list(data_hypo['ID']) if list(data_hypo['ID']).count(x) >= 2]))[0]
    ids_train = ids_test  

    unique_group_list = list(product(ids_train, ids_test))
    
    auroc_goupe = pd.DataFrame(columns=ids_test, index=ids_train).add_prefix('test_')
    auroc_goupe_val = pd.DataFrame(columns=ids_test, index=ids_train).add_prefix('test_')
    
    auroc_goupe.index = auroc_goupe.index.to_series().add_prefix('train_').index
    auroc_goupe_val.index = auroc_goupe_val.index.to_series().add_prefix('train_').index

    print(f'\n Test participants: \n\n', ids_test)
    print(f'\n Model parameters: \n\n', GBDT_PARAMS)
    feat = data.drop(columns=['label', 'ID', 'CGM value [mmol/L]']).columns
    print(f'\n Features (no. {len(feat)}):', *feat, sep=f'\n')

    print('\n', '=' * 80)

    with Parallel(n_jobs=16, verbose=0, max_nbytes=None) as parallel:

        for loho_group in unique_group_list:
            loho_run_test = loho_group[1]
            loho_run_train = loho_group[0]

            if loho_run_test in loho_run_train: #Skip these pairs since otherwise one would has the same train and test data.
                continue

            print(f"Running experiment.... Train on {loho_run_train, loho_run_test} and test on {loho_run_test}")

            LOHO_run = data_hypo.loc[data_hypo.ID == loho_run_test].copy()
            LOHO_run.reset_index(inplace=True)
            LOHO_run.drop(columns=['level_0', 'index'], inplace = True)

            data_run = data[data.ID.isin([loho_run_train, loho_run_test])].copy()
            X = data_run.drop(columns=['label', 'CGM value [mmol/L]', 'ID'])
            y = data_run['label'].astype('int')

            score[(loho_run_train, loho_run_test)] = results.copy()
            tpr[(loho_run_train, loho_run_test)] = pd.DataFrame(columns=np.arange(100))
            precision[(loho_run_train, loho_run_test)] = pd.DataFrame(columns=np.arange(100))

            def run_loho(idx, row, LOHO_run):
                
                # 1st select the tes indexes and ensure that test indexes are only from the test ID
                LOHO_index_test = data_run.loc[(data_run.ID == loho_run_test) & (data_run.index >= row['split_begin']) & (data_run.index <= row['split_end'])]
                LOHO_index_test = LOHO_index_test.drop(LOHO_index_test.loc[LOHO_index_test.ID == loho_run_train].index).index

                # 2nd select one random val hypo event; SEED is fixed by the following idx_test * SEED
                val_temp = LOHO_run.index.values.tolist()
                val_temp.remove(idx)
                np.random.seed(idx*SEED)
                idx_val = np.random.choice(val_temp)
                row_val = LOHO_run.iloc[idx_val]

                LOHO_index_val = data_run.loc[(data_run.ID == loho_run_test) & (data_run.index >= row_val['split_begin']) & (data_run.index <= row_val['split_end'])].index

                # Only difference between loho_pers and loho_gen; Simple since we drop the entire test ID from the dataset
                LOHO_index_train = data_run.loc[data_run.ID != loho_run_test].index

                X_train, X_test, X_val = X.loc[LOHO_index_train], X.loc[LOHO_index_test], X.loc[LOHO_index_val]
                y_train, y_test, y_val = y.loc[LOHO_index_train], y.loc[LOHO_index_test], y.loc[LOHO_index_val]

                if len(X_test) > 0 and np.count_nonzero(y_test) > 0 and len(y_test[y_test == 0]) > 0:

                    if TIME != "only":
                        features = X_train.drop(columns=['time']).columns if TIME else X_train.columns
                        transformer = preprocessing.RobustScaler().fit(X_train[features])
                        X_train.loc[:, features], X_test.loc[:, features], X_val.loc[:, features] = transformer.transform(X_train[features]), transformer.transform(X_test[features]), transformer.transform(X_val[features])

                    #X_train = pd.DataFrame(X_train, columns=transformer.feature_names_in_)
                    #X_test = pd.DataFrame(X_test, columns=transformer.feature_names_in_)
                    #X_val = pd.DataFrame(X_val, columns=transformer.feature_names_in_)

                    classifier = lgb.LGBMClassifier(**GBDT_PARAMS)
                    classifier.fit(X_train, y_train, eval_metric='auc')

                    train_score = metrics.roc_auc_score(y_train, classifier.predict_proba(X_train)[:, 1])
                    test_score = metrics.roc_auc_score(y_test, classifier.predict_proba(X_test)[:, 1])
                    val_score = metrics.roc_auc_score(y_val, classifier.predict_proba(X_val)[:, 1])
                    print(f'Results for {loho_group[1]} ({row["split_begin"]:%d.%m. %H:%M:%S} – {row["split_end"]:%d.%m. %H:%M:%S}): AUROC train {train_score:.2f}, val {val_score:.2f}, test {test_score:.2f}')

                    return evaluate(classifier, X_test, y_test, X_train, y_train, X_val, y_val, setting = "gen", ID = loho_group[1], split_end = row["split_end"], split_begin = row["split_begin"])

                else:
                    print(f'For {loho_group[1]} ({row["split_begin"]:%d.%m. %H:%M:%S} – {row["split_end"]:%d.%m. %H:%M:%S}): We do not have a sufficient amount of data.')

            result = parallel(delayed(run_loho)(idx, row, LOHO_run) for idx, row in LOHO_run.iterrows())

            auroc_goupe.at["train_" + loho_run_train, "test_" + loho_run_test] = np.round(
                pd.DataFrame([i[0] for i in result if i is not None], columns=results.columns)['AUROC'].mean(), 2)
            auroc_goupe_val.at["train_" + loho_run_train, "test_" + loho_run_test] = np.round(
                pd.DataFrame([i[0] for i in result if i is not None], columns=results.columns)['AUROC_val'].mean(), 2)

            score["train_" + loho_run_train, "test_" + loho_run_test] = pd.DataFrame([i[0] for i in result if i is not None], columns=results.columns)
            tpr["train_" + loho_run_train, "test_" + loho_run_test] = pd.DataFrame([i[1] for i in result if i is not None])

            precision["train_" + loho_run_train, "test_" + loho_run_test] = pd.DataFrame([i[2] for i in result if i is not None])
            results["train_" + loho_run_train, "test_" + loho_run_test] = score["train_" + loho_run_train, "test_" + loho_run_test].mean()

            print(f'\nFor train: {loho_run_train, loho_run_test} and test: {loho_run_test} is average val score: {auroc_goupe_val.at["train_"+loho_run_train, "test_"+loho_run_test]} and test score: {auroc_goupe.at["train_"+loho_run_train, "test_"+loho_run_test]}\n')

        auroc_goupe.to_csv(results_dir + "AUROC_Grouping_gen.csv")
        auroc_goupe_val.to_csv(results_dir + "AUROC_Grouping_val_gen.csv")

    return score, tpr, precision


def loho(data, data_hypo):

    score, tpr, precision, shap_values, shap_base, shap_data = {}, {}, {}, {}, {}, {}
    results = pd.DataFrame(columns=['AUROC','AUROC_val', 'AUPRC', 'BACC', 'F1 Score', 'MCC', 'Sensitivity', 'Specificity', 'ID_train', 'ID_test'])
    
    print('\n', '=' * 80)
    ids_test = pd.DataFrame(np.unique([x for x in list(data_hypo['ID']) if list(data_hypo['ID']).count(x) > 2]))[0]
    ids_train = ids_test

    unique_group_list = list(product(ids_train, ids_test))
    
    auroc_goupe = pd.DataFrame(columns=ids_test, index=ids_train).add_prefix('test_')
    auroc_goupe_val = pd.DataFrame(columns=ids_test, index=ids_train).add_prefix('test_')
    
    auroc_goupe.index = auroc_goupe.index.to_series().add_prefix('train_').index
    auroc_goupe_val.index = auroc_goupe_val.index.to_series().add_prefix('train_').index

    print(f'\n Test participants: \n\n', ids_test)
    print(f'\n Model parameters: \n\n', GBDT_PARAMS)
    # feat is just for printing purposes.
    feat = data.drop(columns=['label', 'ID', 'CGM value [mmol/L]']).columns 
    print(f'\n Features (no. {len(feat)}):', *feat, sep=f'\n')
    print('\n', '=' * 80)
    
    with Parallel(n_jobs=16, verbose=0, max_nbytes=None) as parallel:

        for loho_group in unique_group_list:
            loho_run_test = loho_group[1]
            loho_run_train = loho_group[0]

            if loho_run_test in loho_run_train: #Skip these pairs since otherwise one would has the same train and test data.
                continue

            print(f"Running personalized subpopulation models.... Train on {loho_run_train, loho_run_test} and test on {loho_run_test}.")

            LOHO_run = data_hypo.loc[data_hypo.ID == loho_run_test].copy()
            LOHO_run.reset_index(inplace=True)

            data_run = data[data.ID.isin([loho_run_train, loho_run_test])].copy()
            X = data_run.drop(columns=['label', 'CGM value [mmol/L]','ID'])
            y = data_run['label'].astype('int')

            score[(loho_run_train, loho_run_test)] = results.copy()
            tpr[(loho_run_train, loho_run_test)] = pd.DataFrame(columns=np.arange(100))
            precision[(loho_run_train, loho_run_test)] = pd.DataFrame(columns=np.arange(100))

            def run_loho(idx, row, LOHO_run):
                
                #1st select the tes indexes and ensure that test indexes are only from the test ID
                LOHO_index_test = data_run.loc[(data_run.ID == loho_run_test) & (data_run.index >= row['split_begin']) & (data_run.index <= row['split_end'])]
                LOHO_index_test = LOHO_index_test.drop(LOHO_index_test.loc[LOHO_index_test.ID == loho_run_train].index).index

                #2nd select one random val hypo event; SEED is fixed by the following idx_test * SEED
                val_temp = LOHO_run.index.values.tolist()
                val_temp.remove(idx)
                np.random.seed(idx*SEED)
                idx_val = np.random.choice(val_temp)
                row_val = LOHO_run.iloc[idx_val]
                LOHO_index_val = data_run.loc[(data_run.ID == loho_run_test) & (data_run.index >= row_val['split_begin']) & (data_run.index <= row_val['split_end'])].index
                
                #3rd the remaining indexes are used for training the model
                LOHO_index_train = data_run.loc[~((data_run.index.isin(LOHO_index_test.append(LOHO_index_val))) & (data_run.ID == loho_run_test))].index

                assert len(LOHO_index_test.append(LOHO_index_val)) == len(data_run) - len(LOHO_index_train), "We dropped more data than expected."

                #Preparing train, val, and test data
                X_train, X_test, X_val = X.loc[LOHO_index_train], X.loc[LOHO_index_test], X.loc[LOHO_index_val]
                y_train, y_test, y_val = y.loc[LOHO_index_train], y.loc[LOHO_index_test], y.loc[LOHO_index_val]

                if len(X_test) > 0 and np.count_nonzero(y_test) > 0 and len(y_test[y_test==0]) > 0:

                    if TIME != "only":
                        features = X_train.drop(columns=['time']).columns if TIME else X_train.columns
                        transformer = preprocessing.RobustScaler().fit(X_train[features])
                        X_train.loc[:, features], X_test.loc[:, features], X_val.loc[:, features] = transformer.transform(X_train[features]), transformer.transform(X_test[features]), transformer.transform(X_val[features])

                    #X_train = pd.DataFrame(X_train, columns=transformer.feature_names_in_)
                    #X_test = pd.DataFrame(X_test, columns=transformer.feature_names_in_)
                    #X_val = pd.DataFrame(X_val, columns=transformer.feature_names_in_)
                    
                    classifier = lgb.LGBMClassifier(**GBDT_PARAMS)
                    classifier.fit(X_train, y_train, eval_metric='auc')

                    train_score = metrics.roc_auc_score(y_train, classifier.predict_proba(X_train)[:, 1])
                    test_score = metrics.roc_auc_score(y_test, classifier.predict_proba(X_test)[:, 1])
                    val_score = metrics.roc_auc_score(y_val, classifier.predict_proba(X_val)[:, 1])

                    print(f'Results for {loho_group[1]} ({row["split_begin"]:%d.%m. %H:%M:%S} – {row["split_end"]:%d.%m. %H:%M:%S}): AUROC train {train_score:.2f}, val {val_score:.2f}, test {test_score:.2f}')

                    return evaluate(classifier, X_test, y_test, X_train, y_train, X_val, y_val, setting = "pers", ID = loho_group[1], split_end = row["split_end"], split_begin = row["split_begin"])
                else:
                    print(f'For {loho_group[1]} ({row["split_begin"]:%d.%m. %H:%M:%S} – {row["split_end"]:%d.%m. %H:%M:%S}): We do not have a sufficient amount of data.')
            
            result = parallel(delayed(run_loho)(idx, row, LOHO_run) for idx, row in LOHO_run.iterrows())

            auroc_goupe.at["train_"+loho_run_train, "test_"+loho_run_test] = np.round(pd.DataFrame([i[0] for i in result if i is not None], columns=results.columns)['AUROC'].mean(), 2)
            auroc_goupe_val.at["train_"+loho_run_train, "test_"+loho_run_test] = np.round(pd.DataFrame([i[0] for i in result if i is not None], columns=results.columns)['AUROC_val'].mean(), 2)

            score["train_"+loho_run_train, "test_"+loho_run_test] = pd.DataFrame([i[0] for i in result if i is not None], columns=results.columns)
            tpr["train_"+loho_run_train, "test_"+loho_run_test] = pd.DataFrame([i[1] for i in result if i is not None])
            precision["train_"+loho_run_train, "test_"+loho_run_test] = pd.DataFrame([i[2] for i in result if i is not None])
            results["train_"+loho_run_train, "test_"+loho_run_test] = score["train_"+loho_run_train, "test_"+loho_run_test].mean()

            print(f'\nFor train: {loho_run_train, loho_run_test} and test: {loho_run_test} is average val score: {auroc_goupe_val.at["train_"+loho_run_train, "test_"+loho_run_test]} and test score: {auroc_goupe.at["train_"+loho_run_train, "test_"+loho_run_test]}\n')

        auroc_goupe.to_csv(results_dir+"AUROC_Grouping_pers.csv")
        auroc_goupe_val.to_csv(results_dir + "AUROC_Grouping_val_pers.csv")

    return score, tpr, precision

if __name__ == '__main__':
    RELOAD_DATA = True # Wenn man immer die gleiche Spezifikation des Dataloaders hat, kann man sich das neualden sparen.
    REDO_EXPERIMENTS = True
    SHAP = False
    EXTEND_TRAINING_DATA = False
    RUN_CONVENTIONAL = False

    PID = os.getpid()
    DELAY = '0min'
    DATE = dt.datetime.now().strftime("%Y_%m_%d")
    IDS_READ = ['radar-01', 'radar-10', 'radar-12', 'radar-13', 'radar-14', 'radar-15', 'radar-17', 'radar-18', 'radar-19',
                'radar-20', 'radar-21', 'radar-22', 'radar-23', 'radar-24', 'radar-25', 'radar-26', 'radar-27', 'radar-28',
                'radar-29', 'radar-30', 'radar-31', 'radar-32', 'radar-33', 'radar-34', 'radar-35', 'radar-36', 'radar-37',
                'radar-38', 'radar-39', 'radar-40']

    WINDOW_LENGTHS = [60]
    WINDOW_STEPS = [60]
    GLUCOSE_THRESHOLDS = [3.9] # 3.9: Hypoglycemia; 13.9 Hyperglycemia
    CALIBRATIONS =  [False]
    TIMES = [True, "only", False] #"only", False
    SEEDS = [i for i in range(1, 12)]
    KS = [10]

    experiments = []

    for experiment in itertools.product(GLUCOSE_THRESHOLDS, WINDOW_LENGTHS, WINDOW_STEPS, CALIBRATIONS, TIMES, SEEDS, KS):
        experiments.append({
            'GLUCOSE_THRESHOLD': experiment[0],
            'WINDOW_LENGTH': experiment[1],
            'WINDOW_STEP': experiment[2],
            'CALIBRATION': experiment[3],
            'TIME': experiment[4],
            'SEED': experiment[5],
            'K': experiment[6]})

    for experiment in experiments:

        WINDOW_LENGTH = experiment['WINDOW_LENGTH']
        WINDOW_STEP = experiment['WINDOW_STEP']
        GLUCOSE_THRESHOLD = experiment['GLUCOSE_THRESHOLD']
        CALIBRATION = experiment['CALIBRATION']
        TIME = experiment['TIME']
        SEED = experiment['SEED']
        K = experiment['K']
        GBDT_PARAMS = {'objective': 'binary', 'learning_rate': 0.05, 'max_depth': 5, 'max_bin': 50,
                      'num_leaves': 5, 'is_unbalance': True, 'metric': 'auc', 'boost_from_average': False, 'random_seed':SEED,  'n_jobs': 4}

        data_dir, results_dir, parquet_filename, numpy_filename = create_filepath()

        print('=' * 80)
        print(' ' * 20, f'Begin evaluation for {("HYPOGLYCEMIA") if GLUCOSE_THRESHOLD == 3.9 else "HYPERGLYCEMIA"}')
        print(f'\nGlobal parameters are \n   - DELAY: {DELAY}  \n   - WINDOW_LENGTH: {WINDOW_LENGTH}'
              f' \n   - WINDOW_STEP: {WINDOW_STEP}  \n   - GLUCOSE_THRESHOLD: {GLUCOSE_THRESHOLD}'
              f'\n   - CALIBRATION: {CALIBRATION}  \n   - TIME: {TIME}\n   - K: {K}\n   - SEED: {SEED}'
              f'\n   - EXTEND_TRAINING_DATA: {EXTEND_TRAINING_DATA}'
              f'\n   - PID: {PID} \n   - DATA DIR: {data_dir} \n   - RESULTS DIR: {results_dir}')

        print('=' * 80)

        if RELOAD_DATA or not all([os.path.exists(parquet_filename % i) for i in ['bg_data', 'data_hypo', 'data']]):
            print('Reloading data')

            with Parallel(n_jobs=min(1, len(IDS_READ))) as parallel:
                result = parallel(
                    delayed(reader)(FEATURE_DIR, BG_DIR, id, WINDOW_LENGTH, WINDOW_STEP,
                                    DELAY=DELAY, EMPATICA=True, CALIBRATION = CALIBRATION,
                                    GLUCOSE_THRESHOLD = GLUCOSE_THRESHOLD) for id in IDS_READ)

            bg_data = pd.concat([i[0] for i in result])
            data_hypo = pd.concat([i[1] for i in result])
            data = pd.concat([i[2] for i in result])

            #plot_glucose_distribution(results_dir, bg_data)

            print('=' * 80)
            print(f"\nOverview of dysglycemia events...\n")
            for i in result:
                print(f'Participant {i[3][0]} has no. {"hypo" if GLUCOSE_THRESHOLD == 3.9 else "hyper"}:  {i[3][1]} and covered by Empatica/Garmin:  {i[3][2]}')
            print(f'\n', '=' * 80)

            bg_data.to_parquet(parquet_filename % 'bg_data', engine='fastparquet')
            data_hypo.to_parquet(parquet_filename % 'data_hypo', engine='fastparquet')
            data.to_parquet(parquet_filename % 'data', engine='fastparquet')

        else:
            print('Loading processed data...')
            bg_data = pd.read_parquet(parquet_filename % 'bg_data', engine='fastparquet')
            data_hypo = pd.read_parquet(parquet_filename % 'data_hypo', engine='fastparquet')
            data = pd.read_parquet(parquet_filename % 'data', engine='fastparquet')
            print('Loaded processed data')

        len_data, len_bg_data = len(data), len(bg_data)
        data.dropna(inplace=True)
        bg_data.dropna(inplace=True)

        print(f"Dropped {len_data - len(data)} rows in data and {len_bg_data - len(bg_data)} rows in glucose data.")

        features = [c for c in data.columns if c.startswith('HEART_RATE_') or c.startswith('hrv_') or c.startswith('eda_') or c.startswith('ZERO_CROSSING_')]


        filter = ['kurtosis', 'skewness', 'lineintegral', '_n_', 'rms', 'num_samples','sum', 'ptp', 'peaks',  '_SCR_', '_time_', 'deriv_p', 'auc_p_', 'amp_p', '_nni',
                  'hrv_mean_hr', 'hrv_std_hr', 'hrv_mean', 'hrv_std', 'hrv_iqr', 'hrv_iqr_5_95', 'hrv_pct_5', 'hrv_pct_95']
        features = [c for c in features if not len([1 for f in filter if f in c])]
        features.append('hrv_rmssd')
        data = data[features + ['label', 'ID', 'CGM value [mmol/L]']]
        data.drop(columns=['hrv_lfnu', 'hrv_hfnu'], inplace = True)


        if TIME:
            data["time"] = data.index.hour

            if TIME == "only":
                data["time"] = data.index.hour
                data.drop(columns=list(data.drop(columns=['label', 'ID', 'CGM value [mmol/L]', 'time']).columns), inplace = True)

        print('=' * 80)
        print(f"Overview of glucose statistics n={len(data.ID.unique())}...\n")

        print(f'-'*13 + " "*5 + f'* CGM' + " "*4 + f'* Emp/Gar *')

        print(f'Hyperglycemia' + " "*5 + f'* {np.round(len(data.loc[data["CGM value [mmol/L]"] > 10]["CGM value [mmol/L]"]) / len(data)*100, 2)}%' + " "*2 +
              f'* {np.round(len(bg_data.loc[bg_data["CGM value [mmol/L]"] > 10]["CGM value [mmol/L]"]) / len(bg_data)*100, 2)}%  *')

        print(f'Euglycemia' + " "*8 + f'* {np.round(len(data.loc[(data["CGM value [mmol/L]"] <= 10) & (data["CGM value [mmol/L]"] >= 3.9)]["CGM value [mmol/L]"]) / len(data)*100, 2)}%' + " " +
              f'* {np.round(len(bg_data.loc[(bg_data["CGM value [mmol/L]"] <= 10) & (bg_data["CGM value [mmol/L]"] >= 3.9)]["CGM value [mmol/L]"]) / len(bg_data)*100, 2)}%  *')

        print(f'Hypoglycemia' + " "*6 + f'* {np.round(len(data.loc[data["CGM value [mmol/L]"] < 3.9]["CGM value [mmol/L]"]) / len(data)*100, 2)}%' + " "*2 +
              f'* {np.round(len(bg_data.loc[bg_data["CGM value [mmol/L]"] < 3.9]["CGM value [mmol/L]"]) / len(bg_data)*100, 2)}%   *')

        print('=' * 80)


        print('Running experiments...')

        score_loho_gen, tpr_loho_gen, precision_loho_gen = loho_gen(data, data_hypo)
        score_loho_pers, tpr_loho_pers, precision_loho_pers = loho(data, data_hypo)



        if RUN_CONVENTIONAL == True:

            score_pers, tpr_pers, precision_pers = personalized(data)
            score_gene, tpr_gene, precision_gene = generalized(data)
            np.save(numpy_filename % 'score_pers', score_pers)
            np.save(numpy_filename % 'tpr_pers', tpr_pers)
            np.save(numpy_filename % 'precision_pers', precision_pers)

            np.save(numpy_filename % 'score_gene', score_gene)
            np.save(numpy_filename % 'tpr_gene', tpr_gene)
            np.save(numpy_filename % 'precision_gene', precision_gene)

        np.save(numpy_filename % 'score_loho_gen', score_loho_gen)
        np.save(numpy_filename % 'tpr_loho_gen', tpr_loho_gen)
        np.save(numpy_filename % 'precision_loho_gen', precision_loho_gen)

        np.save(numpy_filename % 'score_loho_pers', score_loho_pers)
        np.save(numpy_filename % 'tpr_loho_pers', tpr_loho_pers)
        np.save(numpy_filename % 'precision_loho_pers', precision_loho_pers)

