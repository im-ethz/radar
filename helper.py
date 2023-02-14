import os
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import  glob

def remove_highly_correlated_features(X, features, corr_threshold=0.75):
    corr_matrix = X[features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] >= corr_threshold)]
    print(f'# features before: {len(features)}', end=' | ')
    #X.drop(to_drop, axis=1, inplace=True)
    print(f'after: {len(features) - len(to_drop)}', )
    print('dropped:', to_drop)
    return [x for x in features if x not in to_drop]


def plot_glucose_distribution(results_dir, bg_data):
    palette = {"Moderate hyperglycemia": "orange",
               "Mild hyperglycemia": "moccasin",
               "In range": "g",
               "Mild hypoglycemia": "lightcoral",
               "Moderate hypoglycemia": "red"
               }

    bg_data.loc[(bg_data['CGM value [mmol/L]'] > 13.9) & (
            bg_data['CGM value [mmol/L]'] <= 99.0), 'label'] = "Moderate hyperglycemia"  # hyper
    bg_data.loc[(bg_data['CGM value [mmol/L]'] > 10.0) & (
            bg_data['CGM value [mmol/L]'] <= 13.9), 'label'] = "Mild hyperglycemia"

    bg_data.loc[
        (bg_data['CGM value [mmol/L]'] >= 3.9) & (bg_data['CGM value [mmol/L]'] <= 10), 'label'] = "In range"  # normo

    bg_data.loc[
        (bg_data['CGM value [mmol/L]'] >= 3.0) & (bg_data['CGM value [mmol/L]'] < 3.9), 'label'] = "Mild hypoglycemia"
    bg_data.loc[(bg_data['CGM value [mmol/L]'] >= 0.0) & (
            bg_data['CGM value [mmol/L]'] < 3.0), 'label'] = "Moderate hypoglycemia"

    fig = sns.displot(data=bg_data.drop(columns=['ID']).reset_index(),
                      binwidth=0.75,
                      x="CGM value [mmol/L]",
                      # alpha = 0.6,
                      # common_norm=False,
                      hue='label',
                      palette=palette,
                      multiple="stack",
                      stat="density",
                      element="bars").set(title="Glucose Distribution",
                                          xlabel='Glucose in mmol/L')
    fig.savefig(fname=results_dir + '/glucose_distributiion.pdf',
                facecolor="white",
                bbox_inches='tight',
                dpi=300)

    plt.show()
    plt.close()


def dst_changes(bg_data, ID):
    DST_CHANGES = {
        'radar-06': [pd.to_datetime('2021-03-28 08:29:00'), 'w2s'],
        'radar-08': [pd.to_datetime('2021-03-28 19:57:00'), 'w2s'],
        'radar-09': [pd.to_datetime('2021-03-30 11:49:00'), 'w2s'],
        'radar-10': [pd.to_datetime('2021-03-28 09:06:00'), 'w2s'],
        'radar-28': [pd.to_datetime('2021-11-01 08:09:00'), 's2w'],
        'radar-29': [pd.to_datetime('2021-10-30 20:18:00'), 's2w'],
        'radar-30': [pd.to_datetime('2021-10-31 02:00:00'), 's2w'],
    }

    if ID in DST_CHANGES.keys():
        # Converting times where DST was not considered by the bg device
        # 'Etc/GMT-2' = UTC+2 --> Europe/Zurich with DST (Sommerzeit)
        # 'Etc/GMT-1' = UTC+1 --> Europe/Zurich before DST (Winterzeit)

        if DST_CHANGES[ID][1] == 'w2s':
            dst_change = DST_CHANGES[ID][0]

            # Here before DST (Winterzeit)
            temp_w = bg_data.loc[bg_data.index <= dst_change].copy()
            temp_w.index = temp_w.index - dt.timedelta(hours=1)
            temp_w.index = temp_w.index.tz_localize('UTC')

            temp_s = bg_data.loc[bg_data.index > dst_change].copy()
            temp_s.index = temp_s.index - dt.timedelta(hours=2)
            temp_s.index = temp_s.index.tz_localize('UTC')

            bg_data = temp_w.append(temp_s)

        elif DST_CHANGES[ID][1] == 's2w':
            dst_change = DST_CHANGES[ID][0]

            # Here before DST (Winterzeit)
            # Delete additional 2h because the manual shift causes an overlap
            # of observations instead of a jup
            temp_w = bg_data.loc[bg_data.index > dst_change + dt.timedelta(hours=2)].copy()
            temp_w.index = temp_w.index - dt.timedelta(hours=1)
            temp_w.index = temp_w.index.tz_localize('UTC')

            temp_s = bg_data.loc[bg_data.index <= dst_change - dt.timedelta(hours=2)].copy()
            temp_s.index = temp_s.index - dt.timedelta(hours=2)
            temp_s.index = temp_s.index.tz_localize('UTC')

            bg_data = temp_w.append(temp_s)
            bg_data = bg_data.sort_index()

        else:
            # Here before DST (Winterzeit)
            bg_data.index = bg_data.index - pd.Timedelta(hours=1)
            bg_data.index = bg_data.index.tz_localize('UTC')

    else:
        bg_data.index = bg_data.index.tz_localize('Europe/Zurich').tz_convert('UTC')  # .tz_localize(None)

    return bg_data

def reader(FEATURE_DIR, BG_DIR, ID, WINDOW_LENGTH=300, WINDOW_STEP=300, HYPO_DISTANCE='120min', HYPO_EXTENSION='365D', DELAY='0min',
           EMPATICA=True, GLUCOSE_THRESHOLD = 3.9, CALIBRATION = False):
    # read smartwatch data
    garmin_file = f'{FEATURE_DIR}/merged_features_{ID}_{WINDOW_LENGTH}_{WINDOW_STEP}.parquet.gzip'

    if os.path.exists(garmin_file):
        feature_data = pd.read_parquet(glob.glob(garmin_file)[0])
    else:
        print(f'Problems in loading file: {garmin_file} for {ID}.')

    feature_data.index = feature_data.index.tz_localize('UTC')
    feature_data = feature_data.sort_index()
    feature_data = feature_data.filter(regex='^(ZERO)|(hrv)|(HEART_RATE)', axis=1)  # '^(ZERO)|(hrv)|(HEART_RATE)'

    # read empatica data
    if EMPATICA:
        try:
            empatica_files = glob.glob(f'{FEATURE_DIR}empatica_features_{ID}_{WINDOW_LENGTH}_{WINDOW_STEP}.parquet.gzip')[0]

            empatica_data = pd.read_parquet(empatica_files)
            empatica_data = empatica_data.sort_index()
            empatica_data = empatica_data.clip(lower=0)
            empatica_data = empatica_data.add_prefix("eda_")

            average_empatica_missing = []
            for day in pd.date_range(start=sorted(empatica_data.index)[0].date(),end=sorted(empatica_data.index)[-1].date()):
                temp = empatica_data.loc[(empatica_data.index >= day.tz_localize('UTC')) & (empatica_data.index <= day.tz_localize('UTC') + dt.timedelta(days=1)), ["eda_tonic_mean"]].dropna()
                diff_list = pd.Series([pd.to_datetime(day.tz_localize('UTC'))]).append(pd.Series(temp.index))
                diff_list = diff_list.append(pd.Series([pd.to_datetime(day.tz_localize('UTC') + dt.timedelta(days=1))]))
                average_empatica_missing.append(diff_list.diff().sort_values(ascending=False)[0:5].sum().total_seconds() / (0.24 * 3600))
            average_empatica_missing = average_empatica_missing[1:-1]
            print(f"{ID.capitalize()} misses in average {np.round(np.sum(average_empatica_missing) / len(average_empatica_missing), 2)}% (days={len(average_empatica_missing)}) Empatica data exclusively the first and last visit.")
            del temp, diff_list, average_empatica_missing

            feature_data = pd.merge_asof(feature_data,
                                         empatica_data,
                                         left_index=True,
                                         right_index=True,
                                         direction='nearest',
                                         tolerance=pd.Timedelta(seconds=WINDOW_STEP))

        except Exception as e:
            raise Exception("Error in Empatica loading", e)

    feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
    keep_columns = ~feature_data.columns.isin(feature_data.filter(regex="(min)|(max)|(mfcc)|(ent)|(fd)|(ene)").columns.to_list())
    feature_data = feature_data.loc[:, keep_columns]

    # read CGM data
    bg_files = glob.glob(BG_DIR + ID + '/*.csv')
    bg_data = pd.concat([pd.read_csv(f,
                                     names=['datetime', 'event_type', 'CGM value [mmol/L]'],
                                     skiprows=13,
                                     usecols=[0, 1, 3] if ID == "radar-39" else [1, 2, 7],
                                     encoding='ISO-8859-1' if ID == "radar-37" else 'utf-8',
                                     parse_dates=True,
                                     index_col='datetime') for f in bg_files])

    bg_data = bg_data[~bg_data.index.duplicated(keep='first')]
    bg_data = bg_data.sort_index()

    bg_data = bg_data[bg_data.event_type == 'EGV']
    bg_data = bg_data.drop(columns=['event_type'])

    bg_data = dst_changes(bg_data, ID)

    bg_data['CGM value [mmol/L]'] = bg_data['CGM value [mmol/L]'].replace({'High': np.nan, 'Low': np.nan}).astype(float)
    bg_data = bg_data.interpolate(method='time', limit=12)

    bg_data.index = bg_data.index - pd.to_timedelta(DELAY)

    # create hypo file
    temp = bg_data.copy()
    temp['hypo'] = (bg_data['CGM value [mmol/L]'] < GLUCOSE_THRESHOLD).astype(int)
    temp['diff'] = temp['hypo'].diff()
    temp = temp[temp['diff'] != 0]

    temp['timedelta'] = temp.index - temp.index.to_series().shift(1)
    temp['hypo_begin'] = temp.index.to_series().shift(1)
    temp['hypo_enter'] = temp['CGM value [mmol/L]'].shift(1)

    temp = temp[temp['diff'] == -1]
    temp = temp.loc[temp['timedelta'] >= dt.timedelta(minutes=14.75)]

    temp['hypo_end'] = temp.index
    temp = temp.reset_index(drop=True)

    temp['time_to_prev_hypo'] = temp['hypo_begin'] - temp['hypo_end'].shift(1)

    if len(temp) > 1:
        temp.loc[temp.index[0], 'time_to_prev_hypo'] = pd.to_timedelta(HYPO_DISTANCE) + pd.to_timedelta('1min')
        temp = temp.loc[temp['time_to_prev_hypo'] > pd.to_timedelta(HYPO_DISTANCE)]
        temp.loc[temp.index[0], 'time_to_prev_hypo'] = np.nan

    bg_data = bg_data.reindex(index=bg_data.index.union(feature_data.index))
    bg_data = bg_data.interpolate(method='time', limit=12)

    # create combined feature and bg file
    data = feature_data.copy()
    data['CGM value [mmol/L]'] = bg_data['CGM value [mmol/L]']
    data['label'] = 0
    #data = data.drop(data.filter(regex=("(min)|(max)|(mfcc)|(ent)|(fd)"), axis=1).columns.to_list()
    #                 + ['hrv_mean_nni', 'hrv_median_nni'], axis=1)
    #data = data.dropna()

    if CALIBRATION:

        # calibration feature (CGM value at 05:00)
        tmp_index = pd.date_range(start=bg_data.index[0].floor('D') + pd.Timedelta(hours=5) + pd.to_timedelta(DELAY),
                                  end=bg_data.index[-1].floor('D') + pd.Timedelta(days=1),
                                  freq='1D')

        tmp_values = bg_data.reindex(bg_data.index.union(tmp_index), method='nearest',
                                     tolerance=pd.Timedelta(minutes=15))
        tmp_values = tmp_values.loc[tmp_index, 'CGM value [mmol/L]'].rename('calibration')

        data = pd.merge_asof(data, tmp_values,
                             left_index=True,
                             right_index=True,
                             direction='backward',
                             tolerance=pd.to_timedelta('24h'),
                             allow_exact_matches=True)

    # Prepare LOHO data
    loho_data = pd.DataFrame()

    for idx, row in temp.iterrows():
        if len(data.loc[(data.index >= row['hypo_begin']) & (data.index <= row['hypo_end'])].dropna()) > 0:
            # hypo_end = min(row['hypo_begin'] + pd.to_timedelta('1h'), row['hypo_end'])
            data.loc[(data.index >= row['hypo_begin']) & (data.index <= row['hypo_end']), 'label'] = 1
            data.loc[(data.index > row['hypo_end']) & (data.index <= row['hypo_end'] + pd.to_timedelta(HYPO_DISTANCE)), 'label'] = -1
            loho_data = loho_data.append(row)


    data = data[data['label'] != -1]
    loho_data = loho_data.reset_index()

    if not loho_data.empty:

        loho_data['min_allowed_split_begin'] = loho_data['hypo_begin'] - (
                    loho_data['hypo_begin'] - loho_data['hypo_end'].shift()) / 2
        loho_data['max_allowed_split_end'] = loho_data['hypo_end'] + (
                    loho_data['hypo_begin'].shift(-1) - loho_data['hypo_end']) / 2

        loho_data.loc[loho_data.index[0], 'min_allowed_split_begin'] = data.index[0]
        loho_data.loc[loho_data.index[-1], 'max_allowed_split_end'] = data.index[-1]

        loho_data['split_begin'] = np.maximum(loho_data['min_allowed_split_begin'],
                                             loho_data['hypo_begin'] - pd.to_timedelta(HYPO_EXTENSION))
        loho_data['split_end'] = np.minimum(loho_data['max_allowed_split_end'],
                                           loho_data['hypo_end'] + pd.to_timedelta(HYPO_EXTENSION))

    COMPUTE_QUEST_FEATURES = False
    if COMPUTE_QUEST_FEATURES:
        quest_data = pd.read_csv(ROOT + "questionnaire/questionnaire.csv")
        quest_data = quest_data.loc[quest_data['pC'] == ID]

        data['PA_now'] = 0
        data['PA_now_intensity'] = 0

        data['PA_12h'] = 0
        data['PA_12h_intensity'] = 0

        data['PA_24h'] = 0
        data['PA_24h_intensity'] = 0

        PA_intensity = {"leicht": 1, "mittel": 2, "hoch": 3, "Easy": 1, "Moderate": 2, "High": 3}
        quest_data['Datum Abgeschickt'] = quest_data['Datum Abgeschickt'].astype(str)
        quest_data['Wann haben Sie die Aktivität begonnen?'] = quest_data[
            'Wann haben Sie die Aktivität begonnen?'].astype(str)
        quest_data['Wann haben Sie die Akitvität beendet?'] = quest_data[
            'Wann haben Sie die Akitvität beendet?'].astype(str)

        for idx, row in quest_data.loc[(quest_data['Haben Sie sich gestern körperlich betätigt?'] == "Yes") |
                                       (quest_data['Haben Sie sich gestern körperlich betätigt?'] == "Ja")].iterrows():
            if not row['Datum Abgeschickt'] == "nan":
                intensity = row['Wie schätzen Sie die Intensität Ihrer Aktivität ein?']

                activity_start = pd.to_datetime(row['Datum Abgeschickt'].split(' ')[0] + ' ' +
                                                row['Wann haben Sie die Aktivität begonnen?']
                                                ).tz_localize('Europe/Zurich').tz_convert('UTC') - pd.to_timedelta('1d')

                activity_end = pd.to_datetime(row['Datum Abgeschickt'].split(' ')[0] + ' ' +
                                              row['Wann haben Sie die Akitvität beendet?']
                                              ).tz_localize('Europe/Zurich').tz_convert('UTC') - pd.to_timedelta('1d')

                data.loc[(data.index >= activity_start) & (data.index <= activity_end), 'PA_now'] = 1
                data.loc[(data.index >= activity_start) & (data.index <= activity_end), 'PA_now_intensity'] = \
                PA_intensity[intensity]

                data.loc[
                    (data.index >= activity_end) & (data.index <= activity_end + pd.to_timedelta('12H')), 'PA_12h'] = 1
                data.loc[(data.index >= activity_end) & (
                            data.index <= activity_end + pd.to_timedelta('12H')), 'PA_12h_intensity'] = PA_intensity[
                    intensity]

                data.loc[
                    (data.index >= activity_end) & (data.index <= activity_end + pd.to_timedelta('24H')), 'PA_24h'] = 1
                data.loc[(data.index >= activity_end) & (
                            data.index <= activity_end + pd.to_timedelta('24H')), 'PA_24h_intensity'] = PA_intensity[
                    intensity]

    bg_data['ID'] = ID
    loho_data['ID'] = ID
    data['ID'] = ID

    return bg_data, loho_data, data, [ID, len(temp), len(loho_data)]



def generate_shap_plots(explainer, setting):

    importances = pd.DataFrame(index=explainer.feature_names,
                               data={'importance': explainer[:, :, 1].abs.mean(0).values})
    importances['mod'] = importances.index.str.slice(0, 3)

    mods = list(set([i[:3] for i in explainer.feature_names]))

    pers_importances = [np.round(importances[importances['mod'] == i]['importance'].sum() / importances['importance'].sum(), 4) * 100 for i in mods]
    pers_importances = pd.DataFrame(np.array(pers_importances).reshape(1, len(mods)), columns=mods)
    pers_importances.to_csv(results_dir + f'/shap_importances_{setting}.csv')

    shap.plots.beeswarm(explainer[:, :, 1], max_display=15, plot_size=(14, 12), show=False, alpha=0.3)

    plt.savefig(fname=results_dir + f'/shap_plot_{setting}.pdf',
                facecolor="white",
                bbox_inches='tight',
                dpi=300)
    plt.show()
    plt.close()

    fig, axs_plain = plt.subplots(3, 1, figsize=(8, 12), facecolor='white', constrained_layout=False)
    axs = axs_plain.ravel()

    for idx, mod in enumerate([['hrv', 'HEA'], ['eda'], ['ZER']]):
        ax = axs[idx]
        fig.sca(ax)
        # features = list(filter(lambda x: x.startswith('%s_' % mod.lower()), shap_values.feature_names))
        features = importances[importances['mod'].isin(mod)].sort_values(by='importance', ascending=False)[
                   :5].index.to_list()
        shap.plots.beeswarm(explainer[:, features, 1], max_display=5, plot_size=(10, 12),
                            show=False, alpha=0.3)

        x_max = max(np.abs(ax.get_xlim()))
        ax.set_xlim([-x_max, x_max])
        # break
    plt.subplots_adjust(hspace=0.35)

    fig.savefig(fname=results_dir + f'/shap_plot_modality_{setting}.pdf',
                 facecolor="white",
                 bbox_inches='tight',
                 dpi=300)
    plt.show()
    plt.close()

    return None

def generate_plot_tree(classifier, filepath):
    lgb.plot_tree(classifier, dpi=600)

    plt.savefig(fname=Path(filepath),
                facecolor="white",
                bbox_inches='tight',
                dpi=600)
    plt.show()
    plt.close()
    return None

def plot_curves(tpr_pers, precision_pers, score_pers,
                tpr_loho, precision_loho, score_loho, prc_baseline):

    # Preliminaries summary ROC Curve (x: FPR; y:TPR)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(22, 10))

    fpr = np.linspace(0, 1, 100)
    recall = np.linspace(1, 0, 100)

    #====ROC Plot summary======
    '''
    tpr_gene_mean = pd.DataFrame([tpr_gene[ID][0] for ID in tpr_gene.keys()]).mean()
    tpr_gene_std = pd.DataFrame([tpr_gene[ID][0] for ID in tpr_gene.keys()]).std()

    roc_auc_gene_mean = np.round(np.mean([np.mean(score_gene[ID]['AUROC']) for ID in score_gene.keys()]), 2)
    roc_auc_gene_std = np.round(np.std([np.mean(score_gene[ID]['AUROC']) for ID in score_gene.keys()]), 2)

    ax.plot(fpr, tpr_gene_mean , color='g',label=r'Generalized: Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_auc_gene_mean, roc_auc_gene_std),lw=2)
    ax.fill_between(fpr, np.clip(tpr_gene_mean - tpr_gene_std, a_min=0, a_max=1), np.clip(tpr_gene_mean + tpr_gene_std, a_min=0, a_max=1), color='green', alpha=.2, label=r'$\pm$ 1 std. dev.')
    '''

    tpr_pers_mean = pd.DataFrame([np.mean(tpr_pers[ID]) for ID in tpr_pers.keys()]).mean()
    tpr_pers_std = pd.DataFrame([np.mean(tpr_pers[ID]) for ID in tpr_pers.keys()]).std()

    roc_auc_pers_mean = np.round(np.mean([np.mean(score_pers[ID]['AUROC']) for ID in score_pers.keys()]), 2)
    roc_auc_pers_std = np.round(np.std([np.mean(score_pers[ID]['AUROC']) for ID in score_pers.keys()]), 2)
    ax.plot(fpr, tpr_pers_mean, color='b',label=r'Personalized: Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_auc_pers_mean, roc_auc_pers_std),lw=2)
    ax.fill_between(fpr, np.clip(tpr_pers_mean - tpr_pers_std, a_min=0, a_max=1), np.clip(tpr_pers_mean + tpr_pers_std, a_min=0, a_max=1), color='blue', alpha=.2, label=r'$\pm$ 1 std. dev.')

    #LOHO

    tpr_loho_mean = pd.DataFrame([np.mean(tpr_loho[ID]) for ID in tpr_loho.keys()]).mean()
    tpr_loho_std = pd.DataFrame([np.mean(tpr_loho[ID]) for ID in tpr_loho.keys()]).std()

    roc_auc_loho_mean = np.round(np.mean([np.mean(score_loho[ID]['AUROC']) for ID in score_loho.keys()]), 2)
    roc_auc_loho_std = np.round(np.std([np.mean(score_loho[ID]['AUROC']) for ID in score_loho.keys()]), 2)

    ax.plot(fpr, tpr_loho_mean, color='darkorange',
            label=r'LOHO: Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_auc_loho_mean, roc_auc_loho_std), lw=2)
    ax.fill_between(fpr, np.clip(tpr_loho_mean - tpr_loho_std, a_min=0, a_max=1),
                    np.clip(tpr_loho_mean + tpr_loho_std, a_min=0, a_max=1), color='darkorange', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    #====PRC Plot summary======
    '''
    precision_gene_mean = pd.DataFrame([precision_gene[ID][0] for ID in precision_gene.keys()]).mean()
    precision_gene_std = pd.DataFrame([precision_gene[ID][0] for ID in precision_gene.keys()]).std()

    prc_auc_gene_mean = np.round(np.mean([np.mean(score_gene[ID]['AUPRC']) for ID in score_gene.keys()]), 2)
    prc_auc_gene_std = np.round(np.std([np.mean(score_gene[ID]['AUPRC']) for ID in score_gene.keys()]), 2)

    ax2.plot(recall, precision_gene_mean , color='g',label=r'Generalized: Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (prc_auc_gene_mean, prc_auc_gene_std),lw=2)
    ax2.fill_between(recall, np.clip(precision_gene_mean - precision_gene_std, a_min=0, a_max=1), np.clip(precision_gene_mean + precision_gene_std, a_min=0, a_max=1), color='green', alpha=.2, label=r'$\pm$ 1 std. dev.')
    '''

    precision_pers_mean = pd.DataFrame([np.mean(precision_pers[ID]) for ID in precision_pers.keys()]).mean()
    precision_pers_std = pd.DataFrame([np.mean(precision_pers[ID]) for ID in precision_pers.keys()]).std()

    prc_auc_pers_mean = np.round(np.mean([np.mean(score_pers[ID]['AUPRC']) for ID in score_pers.keys()]), 2)
    prc_auc_pers_std = np.round(np.std([np.mean(score_pers[ID]['AUPRC']) for ID in score_pers.keys()]), 2)

    ax2.plot(recall, precision_pers_mean, color='b',label=r'Personalized: Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (prc_auc_pers_mean, prc_auc_pers_std),lw=2)
    ax2.fill_between(recall, np.clip(precision_pers_mean - precision_pers_std, a_min=0, a_max=1), np.clip(precision_pers_mean + precision_pers_std, a_min=0, a_max=1), color='blue', alpha=.2, label=r'$\pm$ 1 std. dev.')

    # LOHO

    precision_loho_mean = pd.DataFrame([np.mean(precision_loho[ID]) for ID in precision_loho.keys()]).mean()
    precision_loho_std = pd.DataFrame([np.mean(precision_loho[ID]) for ID in precision_loho.keys()]).std()

    prc_auc_loho_mean = np.round(np.mean([np.mean(score_loho[ID]['AUPRC']) for ID in score_loho.keys()]), 2)
    prc_auc_loho_std = np.round(np.std([np.mean(score_loho[ID]['AUPRC']) for ID in score_loho.keys()]), 2)

    ax2.plot(recall, precision_loho_mean, color='darkorange',
             label=r'LOHO: Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (prc_auc_loho_mean, prc_auc_loho_std), lw=2)
    ax2.fill_between(recall, np.clip(precision_loho_mean - precision_loho_std, a_min=0, a_max=1),
                     np.clip(precision_loho_mean + precision_loho_std, a_min=0, a_max=1), color='darkorange', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    #====================
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
    ax.set(xlim=[-0.02, 1.02], ylim=[-0.02, 1.02])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.legend(loc="lower right")
    ax.grid()


    ax2.plot([0, 1], [prc_baseline, prc_baseline], linestyle='--', lw=2, color='r', label='Baseline (%.2f)' % prc_baseline, alpha=.5)
    ax2.set(xlim=[-0.02, 1.02], ylim=[-0.02, 1.02])
    ax2.set_ylabel('Precision')
    ax2.set_xlabel('Recall')
    ax2.legend(loc="upper right")
    ax2.grid()

    fig.savefig(fname=results_dir + '/prcprc_plot.pdf',
                 facecolor="white",
                 bbox_inches='tight',
                 dpi=300)
    plt.show()
    plt.close()

    return None

