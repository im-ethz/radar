import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
import tqdm
import shap

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob as glob
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn import preprocessing
from joblib import Parallel, delayed
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

from shap.plots.colors import red_blue
from helper import reader

#=========================
# Specify the results that you want to read to compute the results
WINDOW_LENGTH = 60
WINDOW_STEP = 60
GLUCOSE_THRESHOLD = 3.9
CALIBRATION = False
TIME = True  # False, time_only
#=========================
# Set global parameters for loading data and results
DELAY = '0min'

ROOT = 'filepath'
FEATURE_DIR = ROOT + 'output/features/'
BG_DIR = ROOT + 'dexcom-entries/'

ROOT_RESULTS = 'filepath'
DATE_OF_EXPERIMENT = 'DATE_OF_EXPERIMENT'
PID = 'PID'

if TIME:
    str_time = "time"
elif TIME == False:
    str_time = "no_time"
elif TIME == "only":
    str_time = "time_only"

RESULTS_DIR = ROOT_RESULTS + f"/results/{DATE_OF_EXPERIMENT}/{PID}/hypo"
#=========================

def calculate_results(seed, threshold, gen, revision):
    mean_scores = pd.DataFrame()
    aurocs, tuples = [], []
    scores, tpr = {}, {}
    all_tpr = pd.DataFrame()

    auroc_goup = pd.read_csv(seed + f"WL60_WS60/wocalibration/{str_time}/AUROC_Grouping_{'gen' if gen else 'pers'}.csv",
                             index_col=0)

    auroc_goupe_val = pd.read_csv(
        seed + f"WL60_WS60/wocalibration/{str_time}/AUROC_Grouping_val_{'gen' if gen else 'pers'}.csv",
        index_col=0)

    score_loho = np.load(seed + f"WL60_WS60/wocalibration/{str_time}/score_loho_{'gen' if gen else 'pers'}.npy",
                         allow_pickle=True).item()

    tpr_loho = np.load(seed + f"WL60_WS60/wocalibration/{str_time}/tpr_loho_{'gen' if gen else 'pers'}.npy",
                       allow_pickle=True).item()

    # pers_precision_loho = np.load(numpy_filename % 'precision_loho', allow_pickle=True).item()

    # pers_auroc_goupe_val_mean = pers_auroc_goupe_val.max().mean()
    selected_max_larger_than_t = auroc_goupe_val.columns[auroc_goupe_val.max() >= threshold]
    # print(selected_max_larger_than_t)

    for col in selected_max_larger_than_t:
        idx = pd.to_numeric(auroc_goupe_val[col]).idxmax(axis=0)
        # print(idx)

        if revision == "PNP":
            # This line is for revision/ only PNP
            ids_revision = ['radar-01', 'radar-10', 'radar-14', 'radar-20', 'radar-34', 'radar-35']

        elif revision == "TBR":
            # This line is for revision/ only Patients with higher hypo frequency
            ids_revision = ['radar-10', 'radar-13', 'radar-34', 'radar-17', 'radar-32', 'radar-30',
                            'radar-26', 'radar-19', 'radar-20', 'radar-22', 'radar-28']

        elif revision == "T1DM":
            # This line is for revision/ only Patients withT1DM
            #ids_type1 = pd.read_csv(ROOT + "/demographics/RADAR_Demographics_Processed.csv", index_col=0)
            #ids_revision = list(ids_type1.loc[ids_type1["diab_type"] == "Typ 1"].pat_id)
            ids_revision = ['radar-10', 'radar-13', 'radar-14', 'radar-15', 'radar-17', 'radar-23', 'radar-26',
                            'radar-30''radar-32', 'radar-35', 'radar-37', 'radar-38', 'radar-39', 'radar-40']

        if revision != "None":
            if col.split("_")[-1] in ids_revision:
                aurocs.append(auroc_goup.loc[idx, col])
                tuples.append((idx, col))

        else:
            aurocs.append(auroc_goup.loc[idx, col])
            tuples.append((idx, col))

    for tuple_ in tuples:
        scores[tuple_[1]] = score_loho[tuple_]
        tpr[tuple_[1]] = tpr_loho[tuple_]

    # for key in pers_lower_bound_tpr.keys():
    #    all_tpr = all_tpr.append(pers_lower_bound_tpr[key])

    for key in scores.keys():
        mean_scores = mean_scores.append(scores[key].iloc[:, 0:8].mean(), ignore_index=True)

    # mean_score = np.round(np.mean([np.mean(pers_lower_bound_scores[ID]['AUROC']) for ID in pers_lower_bound_scores.keys()]), 2)

    return tpr, mean_scores, tuples


def plot_pers_AND_gen(pers_tpr, pers_scores, pers_imp, gen_tpr, gen_scores, gen_imp, ids_pers, ids_gen, revision):
    ## ====== Varianta B ==========

    # Preliminaries summary ROC Curve (x: FPR; y:TPR)
    # fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fpr = np.linspace(0, 1, 100)

    plt.rcParams.update({'font.size': 10})

    f = plt.figure(figsize=(22, 8), facecolor='white')
    gs1 = GridSpec(1, 2, right=2 / 3 - 0.05)
    # gs2 = GridSpec(2, 1, left=2/3 + 0.08)
    ax = f.add_subplot(gs1[0, 0])
    ax2 = f.add_subplot(gs1[0, 1])
    # ax3 = f.add_subplot(gs2[0,0])
    # ax4 = f.add_subplot(gs2[1,0])

    ax.set(aspect='equal', adjustable='box', xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6),
           xlim=[-0.02, 1.02], ylim=[-0.02, 1.02], axisbelow=True)

    ax2.set(aspect='equal', adjustable='box', xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6),
            xlim=[-0.02, 1.02], ylim=[-0.02, 1.02], axisbelow=True)
    # ax3.set(aspect='equal', adjustable='box', xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6), xlim=[-0.02, 1.02], ylim=[-0.02, 1.02], axisbelow=True)
    # ax4.set(aspect='equal', adjustable='box', xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6), xlim=[-0.02, 1.02], ylim=[-0.02, 1.02], axisbelow=True)
    # ====ROC Plot for personalized LOHO ======

    pers_tpr_mean = pd.DataFrame([np.mean(pers_tpr[ID]) for ID in pers_tpr.keys()]).mean()
    pers_tpr_std = pd.DataFrame([np.mean(pers_tpr[ID]) for ID in pers_tpr.keys()]).std()

    pers_auroc_mean = pers_scores.mean().round(3)['AUROC']
    pers_auroc_std = pers_scores.std().round(3)['AUROC']

    ax.plot(fpr, pers_tpr_mean, color='black', label=r'ROC curve of personalized model (AUC = %0.2f $\pm$ %0.2f)' % (
        pers_auroc_mean, pers_auroc_std), lw=2)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=1, color='black', alpha=.8)
    ax.set(xlim=[-0.02, 1.02], ylim=[-0.02, 1.02])
    ax.set_ylabel('Sensitivity')
    ax.set_xlabel('1 - Specificity')
    ax.set_title(r"Personalisiert; Threshold=%0.2f; N=%d/%d" % (threshold, len(pers_scores), len(ids_pers)))
    # ax.grid()

    # Shrink current axis's height by 10% on the bottom and put a legend below current axis
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

    # ax.fill_between(fpr, np.clip(pers_tpr_mean - pers_tpr_std, a_min=0, a_max=1),
    #                np.clip(pers_tpr_mean + pers_tpr_std, a_min=0, a_max=1), color='darkgreen',
    #                alpha=.2,
    #                label=r'$\pm$ 1 std. dev.')
    # ====================

    # ====ROC Plot for generalized LOHO ======
    gen_tpr_mean = pd.DataFrame([np.mean(gen_tpr[ID]) for ID in gen_tpr.keys()]).mean()
    gen_tpr_std = pd.DataFrame([np.mean(gen_tpr[ID]) for ID in gen_tpr.keys()]).std()

    gen_auroc_mean = gen_scores.mean().round(3)['AUROC']
    gen_auroc_std = gen_scores.std().round(3)['AUROC']

    ax2.plot(fpr, gen_tpr_mean, color='black',
             label=r'ROC curve of generalized model (AUC = %0.2f $\pm$ %0.2f)' % (gen_auroc_mean, gen_auroc_std), lw=2)

    # Shrink current axis's height by 10% on the bottom
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

    # ax2.fill_between(fpr, np.clip(gen_tpr_mean - gen_tpr_std, a_min=0, a_max=1),
    #                 np.clip(gen_tpr_mean + gen_tpr_std, a_min=0, a_max=1), color='darkorange',
    #                 alpha=.2,
    #                 label=r'$\pm$ 1 std. dev.')

    ax2.plot([0, 1], [0, 1], linestyle='--', lw=1, color='black', alpha=.8)
    ax2.set(xlim=[-0.02, 1.02], ylim=[-0.02, 1.02])
    ax2.set_ylabel('Sensitivity')
    ax2.set_xlabel('1 - Specificity')
    ax2.set_title(r"Generalisiert; Threshold=%0.2f; N=%d/%d" % (threshold, len(gen_scores), len(ids_gen)))
    '''
    # ====================
    pers_imp_unstacked = pers_imp.unstack().to_frame()
    sns.barplot(y = pers_imp_unstacked.index.get_level_values(0), x = pers_imp_unstacked[0], ci = None, color = "black", ax = ax3, label='Feature Importance (SHAP)')
    box = ax3.get_position()
    ax3.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.8])
    #ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
    ax3.set_title("Feature Importance")

    gen_imp_unstacked = gen_imp.unstack().to_frame()

    sns.barplot(y = gen_imp_unstacked.index.get_level_values(0), x = gen_imp_unstacked[0], ci = None, color = "black", ax = ax4, label='Feature Importance (SHAP)')
    box = ax4.get_position()
    ax4.set_position([box.x0, box.y0 + box.height * 0.05, box.width, box.height * 0.8])
    ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
    #ax4.set_title("Feature Importance")
    '''
    f.savefig(fname=RESULTS_DIR + f'/roc_variante_{str_time}_{revision}.pdf', facecolor="white", bbox_inches='tight',
              dpi=300)

    plt.show()

    # plt.close()

def plot_pers_AND_gen(pers_tpr, pers_scores, pers_imp, gen_tpr, gen_scores, gen_imp, ids_pers, ids_gen, revision, threshold):
    ## ====== Varianta B ==========

    # Preliminaries summary ROC Curve (x: FPR; y:TPR)
    # fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fpr = np.linspace(0, 1, 100)

    plt.rcParams.update({'font.size': 10})

    f = plt.figure(figsize=(22, 8), facecolor='white')
    gs1 = GridSpec(1, 2, right=2 / 3 - 0.05)
    # gs2 = GridSpec(2, 1, left=2/3 + 0.08)
    ax = f.add_subplot(gs1[0, 0])
    ax2 = f.add_subplot(gs1[0, 1])
    # ax3 = f.add_subplot(gs2[0,0])
    # ax4 = f.add_subplot(gs2[1,0])

    ax.set(aspect='equal', adjustable='box', xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6),
           xlim=[-0.02, 1.02], ylim=[-0.02, 1.02], axisbelow=True)

    ax2.set(aspect='equal', adjustable='box', xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6),
            xlim=[-0.02, 1.02], ylim=[-0.02, 1.02], axisbelow=True)
    # ax3.set(aspect='equal', adjustable='box', xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6), xlim=[-0.02, 1.02], ylim=[-0.02, 1.02], axisbelow=True)
    # ax4.set(aspect='equal', adjustable='box', xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6), xlim=[-0.02, 1.02], ylim=[-0.02, 1.02], axisbelow=True)
    # ====ROC Plot for personalized LOHO ======

    pers_tpr_mean = pd.DataFrame([np.mean(pers_tpr[ID]) for ID in pers_tpr.keys()]).mean()
    pers_tpr_std = pd.DataFrame([np.mean(pers_tpr[ID]) for ID in pers_tpr.keys()]).std()

    pers_auroc_mean = pers_scores.mean().round(3)['AUROC']
    pers_auroc_std = pers_scores.std().round(3)['AUROC']

    ax.plot(fpr, pers_tpr_mean, color='black',
            label=r'ROC curve of personalized model (AUC = %0.2f $\pm$ %0.2f)' % (
                pers_auroc_mean, pers_auroc_std), lw=2)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=1, color='black', alpha=.8)
    ax.set(xlim=[-0.02, 1.02], ylim=[-0.02, 1.02])
    ax.set_ylabel('Sensitivity')
    ax.set_xlabel('1 - Specificity')
    ax.set_title(r"Personalisiert; Threshold=%0.2f; N=%d/%d" % (threshold, len(pers_scores), len(ids_pers)))
    # ax.grid()

    # Shrink current axis's height by 10% on the bottom and put a legend below current axis
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

    # ax.fill_between(fpr, np.clip(pers_tpr_mean - pers_tpr_std, a_min=0, a_max=1),
    #                np.clip(pers_tpr_mean + pers_tpr_std, a_min=0, a_max=1), color='darkgreen',
    #                alpha=.2,
    #                label=r'$\pm$ 1 std. dev.')
    # ====================

    # ====ROC Plot for generalized LOHO ======
    gen_tpr_mean = pd.DataFrame([np.mean(gen_tpr[ID]) for ID in gen_tpr.keys()]).mean()
    gen_tpr_std = pd.DataFrame([np.mean(gen_tpr[ID]) for ID in gen_tpr.keys()]).std()

    gen_auroc_mean = gen_scores.mean().round(3)['AUROC']
    gen_auroc_std = gen_scores.std().round(3)['AUROC']

    ax2.plot(fpr, gen_tpr_mean, color='black',
             label=r'ROC curve of generalized model (AUC = %0.2f $\pm$ %0.2f)' % (gen_auroc_mean, gen_auroc_std),
             lw=2)

    # Shrink current axis's height by 10% on the bottom
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

    # ax2.fill_between(fpr, np.clip(gen_tpr_mean - gen_tpr_std, a_min=0, a_max=1),
    #                 np.clip(gen_tpr_mean + gen_tpr_std, a_min=0, a_max=1), color='darkorange',
    #                 alpha=.2,
    #                 label=r'$\pm$ 1 std. dev.')

    ax2.plot([0, 1], [0, 1], linestyle='--', lw=1, color='black', alpha=.8)
    ax2.set(xlim=[-0.02, 1.02], ylim=[-0.02, 1.02])
    ax2.set_ylabel('Sensitivity')
    ax2.set_xlabel('1 - Specificity')
    ax2.set_title(r"Generalisiert; Threshold=%0.2f; N=%d/%d" % (threshold, len(gen_scores), len(ids_gen)))
    '''
    # ====================
    pers_imp_unstacked = pers_imp.unstack().to_frame()
    sns.barplot(y = pers_imp_unstacked.index.get_level_values(0), x = pers_imp_unstacked[0], ci = None, color = "black", ax = ax3, label='Feature Importance (SHAP)')
    box = ax3.get_position()
    ax3.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.8])
    #ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
    ax3.set_title("Feature Importance")

    gen_imp_unstacked = gen_imp.unstack().to_frame()

    sns.barplot(y = gen_imp_unstacked.index.get_level_values(0), x = gen_imp_unstacked[0], ci = None, color = "black", ax = ax4, label='Feature Importance (SHAP)')
    box = ax4.get_position()
    ax4.set_position([box.x0, box.y0 + box.height * 0.05, box.width, box.height * 0.8])
    ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
    #ax4.set_title("Feature Importance")
    '''
    f.savefig(fname=RESULTS_DIR + f'/roc_variante_{str_time}_{revision}.pdf', facecolor="white",
              bbox_inches='tight', dpi=300)

    plt.show()

    # plt.close()


def read_shap(seed, gen):
    shap_dir = seed + "WL60_WS60/wocalibration/time/"
    imp, raw_values, raw_base_values, raw_data = pd.DataFrame(), [], [], []

    for i in glob.glob(shap_dir + f"/**/sh_imp_{'gen' if gen else 'pers'}_**.csv", recursive=True):
        imp = imp.append(
            pd.read_csv(i, names=['feature', 'importance'], index_col='feature', skiprows=1).sort_index().transpose())

    imp = imp[imp.mean().sort_values(ascending=False)[:10].index]
    read_raw = False
    raw = None
    if read_raw:
        for i in glob.glob(shap_dir + f"/**/sh_raw_{'gen' if gen else 'pers'}_**.pkl", recursive=True):
            temp_shap_explainer = np.load(i, allow_pickle=True)[:1000, :, 1]
            raw_values.append(temp_shap_explainer.values)
            raw_base_values.append(temp_shap_explainer.base_values)
            raw_data.append(temp_shap_explainer.data)

        raw = np.load(i, allow_pickle=True)[:1000, :, 1]
        raw.values = np.vstack(raw_values)
        raw.base_values = np.vstack(raw_base_values)
        raw.data = np.vstack(raw_data)

    return imp, raw

def calculate_median_results(ids_pers, ids_gen):

    pers_scores, pers_tpr, pers_imp, pers_raw, pers_tuples = {}, {}, {}, {}, {}
    gen_scores, gen_tpr, gen_imp, gen_raw, gen_tuples = {}, {}, {}, {}, {}

    pers_scores_mean, gen_scores_mean = pd.DataFrame(), pd.DataFrame()
    N_pers_lb, N_pers_ub = [], []

    threshold = 0.7
    revision = "None"

    for seed in tqdm.tqdm(sorted(glob.glob(RESULTS_DIR + "/**/"))):
        seed_id = int(seed.split("/")[-2])

        pers_tpr[seed_id], pers_scores[seed_id], pers_tuples[seed_id] = calculate_results(seed, threshold, gen=False,
                                                                                          revision=revision)
        pers_imp[seed_id], pers_raw[seed_id] = read_shap(seed, gen=False)

        gen_tpr[seed_id], gen_scores[seed_id], gen_tuples[seed_id] = calculate_results(seed, threshold, gen=True,
                                                                                       revision=revision)
        gen_imp[seed_id], gen_raw[seed_id] = read_shap(seed, gen=True)

        pers_scores_mean.at[seed_id, "mean_auroc"] = pers_scores[seed_id].mean().round(5)['AUROC']
        gen_scores_mean.at[seed_id, "mean_auroc"] = gen_scores[seed_id].mean().round(5)['AUROC']

        pers_scores_mean.at[seed_id, "No. Participants"] = len(pers_tuples[seed_id])
        gen_scores_mean.at[seed_id, "No. Participants"] = len(gen_tuples[seed_id])

    pers_median = pers_scores_mean.median()['mean_auroc']
    gen_median = gen_scores_mean.median()['mean_auroc']

    pers_median_seed = pers_scores_mean[pers_scores_mean['mean_auroc'] == pers_median].index[0]
    gen_median_seed = gen_scores_mean[gen_scores_mean['mean_auroc'] == gen_median].index[0]

    print(
        f"\n Overall the personalized median auroc is {pers_median} and chose seed {pers_median_seed} and the generalized {gen_median} and seed {gen_median_seed}.")

    if revision == "None":
        plot_pers_AND_gen(pers_tpr[pers_median_seed], pers_scores[pers_median_seed], pers_imp[pers_median_seed],
                          gen_tpr[gen_median_seed], gen_scores[gen_median_seed], gen_imp[gen_median_seed], ids_pers, ids_gen,
                          revision=revision, threshold = threshold)

        print("Pärchen für das generalisierte Model:\n")
        for j in gen_tuples[gen_median_seed]:
            print(j)
    else:
        #To consider the same seed for the sub-analysis, we set the seed manually to the median seed
        #Therefore, the clean analysis has to run first to calculate this seed
        gen_median_seed = 6
        plot_pers_AND_gen(pers_tpr[pers_median_seed], pers_scores[pers_median_seed], pers_imp[pers_median_seed],
                          gen_tpr[gen_median_seed], gen_scores[gen_median_seed], gen_imp[gen_median_seed], ids_pers,
                          ids_gen, revision=revision, threshold=threshold)

    return gen_median_seed, gen_scores, gen_tuples, gen_tpr, gen_scores_mean

def create_final_results(gen_median_seed, gen_scores, gen_tuples, gen_tpr, gen_scores_mean):
    gbdt_params_binary = {'objective': 'binary', 'learning_rate': 0.05, 'max_depth': 5, 'max_bin': 50,
                          'num_leaves': 5, 'is_unbalance': True, 'metric': 'auc', 'random_seed': gen_median_seed,
                          'boost_from_average': False, 'n_jobs': 4, 'first_metric_only': True}
    
    mapper = {"HEART_RATE_mean": "CF_HR mean", "HEART_RATE_std": "CF_HR STD",
              "HEART_RATE_iqr_5_95": r"CF_HR IQR$_{5-95}$", "HEART_RATE_pct_5": r"CF_HR P$_{5\%}$",
              "HEART_RATE_std": "CF_HR STD", "HEART_RATE_pct_95": r"CF_HR P$_{95\%}$",
              "HEART_RATE_iqr": r"CF_HR IQR",
              #
              "ZERO_CROSSING_mean": "MO mean", "ZERO_CROSSING_std": "MO STD",
              "ZERO_CROSSING_iqr_5_95": r"MO IQR$_{5-95}$", "ZERO_CROSSING_pct_5": r"MO P$_{5\%}$",
              "ZERO_CROSSING_pct_95": "MO P$_{95\%}$", "ZERO_CROSSING_iqr": r"MO IQR",
              #
              "hrv_sdsd": "CF_SDSD", "hrv_pnni_50": r"CF_PNNI$_{50}$", "hrv_pnni_20": r"CF_PNNI$_{20}$",
              "hrv_cvsd": "CF_CVSD", "hrv_sdnn": "CF_SDNN", "hrv_cvnni": "CF_CVNNI",
              "hrv_total_power": "CF_Total power", "hrv_vlf": "CF_VLF", "hrv_lf": "CF_LF",
              "hrv_hf": "CF_HF", "hrv_lf_hf_ratio": "CF_LF/HF ratio", "hrv_rmssd": "CF_RMSSD",
              #
              "eda_phasic_mean": "EDA_Phasic mean", "eda_phasic_std": "EDA_Phasic STD",
              "eda_phasic_iqr_5_95": r"EDA_Phasic IQR$_{5-95}$", "eda_phasic_pct_5": r"EDA_Phasic P$_{5\%}$",
              "eda_phasic_pct_95": "EDA_Phasic P$_{95\%}$", "eda_phasic_iqr": r"EDA_Phasic IQR",
              #
              "eda_tonic_mean": "EDA_Tonic mean", "eda_tonic_std": "EDA_Tonic STD",
              "eda_tonic_iqr_5_95": r"EDA_Tonic IQR$_{5-95}$", "eda_tonic_pct_5": r"EDA_Tonic P$_{5\%}$",
              "eda_tonic_pct_95": "EDA_Tonic P$_{95\%}$", "eda_tonic_iqr": r"EDA_Tonic IQR",
              #
              }
    
    
    
    for seed in tqdm.tqdm([gen_median_seed]):  # [gen_median_seed]:#tqdm.tqdm(gen_tuples.keys()):
        ids_seed = []
        for i in gen_tuples[seed]:
            ids_seed.append(i[0].split('_')[-1])
        ids_seed = np.unique(ids_seed)
    
        data_select, data_seed = pd.DataFrame(), pd.DataFrame()
    
        data_seed = data.loc[data['ID'].isin(ids_seed)].copy()
    
        features = [c for c in data_seed.columns if
                    c.startswith('HEART_RATE_') or c.startswith('hrv_') or c.startswith('eda_') or c.startswith(
                        'ZERO_CROSSING_')]
    
        filter = ['kurtosis', 'skewness', 'lineintegral', '_n_', 'rms', 'num_samples', 'sum', 'ptp', 'peaks', '_SCR_',
                  '_time_', 'deriv_p', 'auc_p_', 'amp_p', '_nni',
                  'hrv_mean_hr', 'hrv_std_hr', 'hrv_mean', 'hrv_std', 'hrv_iqr', 'hrv_iqr_5_95', 'hrv_pct_5', 'hrv_pct_95']
    
        features = [c for c in features if not len([1 for f in filter if f in c])]
        features.append('hrv_rmssd')
    
        data_select = data_seed[features + ['label', 'ID', 'CGM value [mmol/L]']].copy()
    
        if TIME:
            data_select['Time'] = data_select.index.hour
    
        data_select.drop(columns=['hrv_lfnu', 'hrv_hfnu'], inplace=True)
    
        features = data_select.drop(columns=['label', 'ID', 'CGM value [mmol/L]']).columns
    
        if TIME:
            list(features).append("Time")
    
        data_select.loc[:, features] = preprocessing.RobustScaler().fit_transform(data_select[features])
    
        X = data_select.drop(columns=['label', 'ID', 'CGM value [mmol/L]'])  # [:40]
        y = data_select['label'].astype('int')  # [:40]
    
        classifier = lgb.LGBMClassifier(**gbdt_params_binary)
        classifier.fit(X, y, eval_metric='auc')
    
        explainer = shap.TreeExplainer(classifier)
        explainer_values = explainer(X.rename(columns=mapper))
        importance = key_display_item(gen_tpr[seed], gen_scores[seed], explainer_values, seed)
    
        for mod in importance.columns:
            gen_scores_mean.at[seed, mod] = importance.loc[0, mod]


def key_display_item(gen_tpr, gen_scores, explainer_values, seed="Median"):
    cmap = plt.get_cmap('inferno', lut=None)
    cmap = ListedColormap([cmap(x) for x in np.linspace(0.2, 0.8, 100)])
    # cmap = red_blue

    fpr = np.linspace(0, 1, 100)

    plt.rcParams.update({'font.size': 10})

    f = plt.figure(figsize=(10, 6), facecolor='white')
    gs1 = GridSpec(2, 1, right=1 / 2 - 0.07, height_ratios=[4, 1])
    # gs2 = GridSpec(2, 1, right=2/3-0.065, left=1/3+0.115, height_ratios=[1, 3])
    gs3 = GridSpec(3, 1, left=1 / 2 + 0.1)
    ax = f.add_subplot(gs1[0, 0])
    ax2 = f.add_subplot(gs1[1, 0])
    # ax3 = f.add_subplot(gs2[1,0])

    gen_tpr_mean = pd.DataFrame([np.mean(gen_tpr[ID]) for ID in gen_tpr.keys()]).mean()
    gen_tpr_std = pd.DataFrame([np.mean(gen_tpr[ID]) for ID in gen_tpr.keys()]).std()

    gen_auroc_mean = gen_scores.mean().round(3)['AUROC']
    gen_auroc_std = gen_scores.std().round(3)['AUROC']

    ax.plot(fpr, gen_tpr_mean, color='black',
            label='ROC curve (AUC = %0.2f $\pm$ %0.2f)' % (gen_auroc_mean, gen_auroc_std), lw=2)
    ax.set_xlim(right=640)

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
    ax.legend(loc='lower right', fancybox=True, shadow=True, ncol=2)
    ax.text(-0.1, 1.05, "A", fontsize='14', weight='bold')
    # ax.text(0.4, -0.2, desc[idx], transform=ax.transAxes, fontsize='14', weight='bold')

    ax.plot([0, 1], [0, 1], linestyle='--', lw=1, color='black', alpha=.8)
    ax.set(xlim=[-0.02, 1.02], ylim=[-0.02, 1.02])
    ax.set_ylabel('Sensitivity')
    ax.set_xlabel('1 - Specificity')

    importances = pd.DataFrame(index=explainer_values.feature_names,
                               data={'importance': explainer_values[:, :, 1].abs.mean(0).values})
    importances['mod'] = importances.index.str.slice(0, 3)

    mods = list(set([i[:3] for i in explainer_values.feature_names]))

    importance_perc = [
        np.round(importances[importances['mod'] == i]['importance'].sum() / importances['importance'].sum(), 4) * 100
        for i in mods]
    importance_perc = pd.DataFrame(np.array(importance_perc).reshape(1, len(mods)), columns=mods)

    importance_perc = importance_perc.rename(columns={"CF_": "CF",
                                                      "Tim": "Time",
                                                      "MO ": "MO",
                                                      "EDA": "EDA"})

    ax.text(-0.1, -0.18, "B", fontsize='14', weight='bold')
    # ax.text(1.1, 0.65, "C", fontsize='14', weight='bold')
    importance_perc = pd.DataFrame(importance_perc.iloc[0].sort_values(ascending=False)).transpose()
    # print(importance_perc.head())
    desc = "CDE"
    for idx, mod in enumerate([['CF_'], ['EDA'], ['MO ']]):

        ax = f.add_subplot(gs3[idx, 0])
        f.sca(ax)
        # features = list(filter(lambda x: x.startswith('%s_' % mod.lower()), shap_values.feature_names))
        features = importances[importances['mod'].isin(mod)].sort_values(by='importance',
                                                                         ascending=False)[:5].index.to_list()
        temp = explainer_values[:, features, 1]
        temp.feature_names = [i.removeprefix("CF_") for i in temp.feature_names]
        temp.feature_names = [i.removeprefix("EDA_") for i in temp.feature_names]

        shap.plots.beeswarm(temp, max_display=5, show=False, color_bar=False, log_scale=True, plot_size=None,
                            color=cmap, alpha=0.5)
        del temp

        x_max = np.round(max(np.abs(ax.get_xlim())), 2)

        ax.set_xlim([-x_max, x_max])
        ax.set_xlim([-1, 1])

        if idx == 2:
            ax.set_xlabel('SHAP Value', labelpad=-10)
            xticklabels = list(map(lambda x: str(x), ax.get_xticks()))
            xticklabels[0] = f'{xticklabels[0]}\nnon-hypoglycemia'
            xticklabels[-1] = f'{xticklabels[-1]}\nhypoglycemia'

            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(xticklabels)

        else:
            ax.get_xaxis().set_visible(False)
            ax.spines['bottom'].set_visible(False)

        ax.text(-0.45, 1.05, desc[idx], transform=ax.transAxes, fontsize='14', weight='bold')

    plt.subplots_adjust(hspace=0.3)

    # fig.tight_layout()
    # fig.subplots_adjust(bottom=0)
    cax = plt.axes([0.97, 0.1, 0.003, 0.8])
    # plt.colorbar(cax=cax)

    # fig.subplots_adjust(bottom=0.12)

    m = cm.ScalarMappable(cmap=cmap)
    m.set_array([0, 1])
    cb = plt.colorbar(m, ticks=[0, 1], aspect=10, orientation='vertical', cax=cax)
    cb.set_ticklabels(['Low', 'High'])
    cb.set_label('Feature Values', size=12, labelpad=-20)
    cb.ax.tick_params(labelsize=11, length=0)
    cb.set_alpha(1)
    cb.outline.set_visible(False)
    # bbox = cb.ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
    # cb.ax.set_aspect((bbox.width - 0.9) * 20)
    cb.draw_all()

    sns.barplot(data=importance_perc, color="black", orient='h', ax=ax2)
    ax2.set_xlabel('Relative importance [%]')
    patches = ax2.patches
    percentage = importance_perc.values[0]
    for i in range(len(patches)):
        x = patches[i].get_width() - patches[i].get_width() / 2
        y = patches[i].get_y() + .6

        ax2.annotate('{:.1f}%'.format(percentage[i]), (x, y), fontsize=9, weight='bold', ha='center', color="white")
        # ax2.set_xlim([0.001, 50])

    names = [i.removeprefix("CF_") for i in explainer_values.feature_names]
    names = [i.removeprefix("EDA_") for i in names]

    # importances = pd.DataFrame(index=names, data={'importance': explainer_values[:, :, 1].abs.mean(0).values})
    # importances.sort_values(by = 'importance', ascending = False, inplace = True)
    # sns.barplot(x = importances['importance'][:15].values, y = importances['importance'][:15].index,
    #            ci = None, color = "black", label='Feature Importance (SHAP)', ax =ax3)
    # ax3.set_xlabel('Mean (|SHAP value|)')

    # ax.set_title(r"Generalisiert; Threshold=%0.2f; N=%d/%d"%(threshold,len(gen_scores), 17))
    f.savefig(fname=RESULTS_DIR + f'/TEST_key_display_{seed}_{str_time}.pdf', facecolor="white", bbox_inches='tight',dpi=300)
    return importance_perc

def glucose_time_statistics(data, ids_gen):
    data_cgm_analysis = data.loc[data['ID'].isin(ids_gen)].copy()

    total_cgm_points = len(data.loc[data['ID'].isin(ids_gen)]['CGM value [mmol/L]']) // 5

    cgm_points_hypo = len(data_cgm_analysis.loc[data_cgm_analysis['CGM value [mmol/L]'] < 3.9]) // 5

    cgm_points_hyper = len(data_cgm_analysis.loc[data_cgm_analysis['CGM value [mmol/L]'] > 10]) // 5

    cgm_points_hypo_perc = np.round(cgm_points_hypo * 100 / total_cgm_points, 3)
    cgm_points_hyper_perc = np.round(cgm_points_hyper * 100 / total_cgm_points, 3)

    print(
        f"In total, we gathered {total_cgm_points} CGM data points covered with both wearables of which {100 - cgm_points_hypo_perc - cgm_points_hyper_perc}% were in euglycemia (3.9–10.0mmol/L), {cgm_points_hypo_perc}% in hypoglycemia (<3.9mmol/L), and {cgm_points_hyper_perc}% in hyperglycemia (>10.0mmol/L).")
    print("\n")
    cgm_hypo_events = []
    for i in result:
        if i[3][0] in list(ids_gen):
            cgm_hypo_events.append(i[3][-1])

    print(f"Number of hypoglycemic events during the study period covered with data from both wearables: {np.sum(cgm_hypo_events)}.")

    data_cgm_analysis['Time'] = data_cgm_analysis.index.hour
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    sns.histplot(data_cgm_analysis.loc[data_cgm_analysis.label == 0].reset_index(), color="darkblue", x="Time", bins=24,
                 stat="density", ax=ax2)
    sns.histplot(data_cgm_analysis.loc[data_cgm_analysis.label == 1].reset_index(), color="darkorange", x="Time",
                 bins=24, stat="density", ax=ax1)

    ax1.set_title("Distribution of feature Time for non-hypoglycemia.")
    ax2.set_title("Distribution of feature Time for hypoglycemia.")

    plt.show()

    return None

if __name__ == '__main__':
    IDS_READ = ['radar-01', 'radar-10', 'radar-12', 'radar-13', 'radar-14', 'radar-15', 'radar-17', 'radar-18',
                'radar-19', 'radar-20', 'radar-21', 'radar-22', 'radar-23', 'radar-24', 'radar-25', 'radar-26',
                'radar-27', 'radar-28', 'radar-29', 'radar-30', 'radar-31', 'radar-32', 'radar-33', 'radar-34',
                'radar-35', 'radar-36', 'radar-37', 'radar-38', 'radar-39', 'radar-40']

    with Parallel(n_jobs=min(4, len(IDS_READ))) as parallel:
        result = parallel(delayed(reader)(FEATURE_DIR, BG_DIR, id, WINDOW_LENGTH, WINDOW_STEP,
                                          DELAY=DELAY, EMPATICA=True, CALIBRATION=CALIBRATION,
                                          GLUCOSE_THRESHOLD=GLUCOSE_THRESHOLD) for id in IDS_READ)
    bg_data = pd.concat([i[0] for i in result])
    data_hypo = pd.concat([i[1] for i in result])
    data = pd.concat([i[2] for i in result])
    data.dropna(inplace=True)

    ids_gen = pd.DataFrame(np.unique([x for x in list(data_hypo['ID']) if list(data_hypo['ID']).count(x) >= 2]))[0]
    ids_pers = pd.DataFrame(np.unique([x for x in list(data_hypo['ID']) if list(data_hypo['ID']).count(x) > 2]))[0]

    glucose_time_statistics(data=data, ids_gen=ids_gen)

    gen_median_seed, gen_scores, gen_tuples, gen_tpr, gen_scores_mean = calculate_median_results(ids_pers=ids_pers, ids_gen=ids_gen)
    
    create_final_results(gen_median_seed=gen_median_seed, gen_scores=gen_scores,
                         gen_tuples=gen_tuples, gen_tpr=gen_tpr, gen_scores_mean=gen_scores_mean)