from helper_model_cte import *
from scipy.spatial.distance import cdist, pdist, squareform
from collections import Counter
from sklearn.model_selection import train_test_split, ParameterSampler
import h2o
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import argparse
import sys
import os


def getArguments():
    parser = argparse.ArgumentParser(description="Running KNN_model for datasets.")
    parser.add_argument("-i1", "--input", dest="inputFile", required=True)
    parser.add_argument("-wif", "--invitroFile", dest="invitroFile", default="False")
    parser.add_argument("-e", "--encoding", dest="encoding", default="binary")
    parser.add_argument(
        "-il", "--invitro_label", dest="invitro_label", default="number"
    )
    parser.add_argument("-dbi", "--db_invitro", dest="db_invitro", default="noinvitro")
    parser.add_argument("-wi", "--w_invitro", dest="w_invitro", default="False")
    parser.add_argument("-ah", "--alpha_h", dest="alpha_h", required=True, nargs="?")
    parser.add_argument("-ap", "--alpha_p", dest="alpha_p", required=True, nargs="?")
    parser.add_argument(
        "-n", "--n_neighbors", dest="n_neighbors", nargs="?", default=1, type=int
    )
    parser.add_argument("-o", "--output", dest="outputFile", default="binary.txt")
    return parser.parse_args()


args = getArguments()
if args.encoding == "binary":
    encoding = "binary"
    encoding_value = 1
elif args.encoding == "multiclass":
    encoding = "multiclass"
    encoding_value = [0.1, 1, 10, 100]


if args.invitroFile == "True":
    categorical = ["class", "tax_order", "family", "genus", "species"]

db_invitro_matrix = None

rand = random.randrange(1, 100)
conc_column = "conc1_mean"
db_mortality = load_data(
    args.inputFile,
    encoding=encoding,
    categorical_columns=categorical,
    conc_column=conc_column,
    encoding_value=encoding_value,
    seed=rand,
)
print("finish loaded.", ctime())
# db_mortality = db_mortality[:300]

test_size = 0.2
col_groups = "test_cas"

df_fishchem = db_mortality[["fish", "test_cas"]]
trainvalid_idx, valid_idx = get_grouped_train_test_split(
    df_fishchem, test_size, col_groups
)
df_fishchem_tv = df_fishchem.iloc[trainvalid_idx, :].reset_index(drop=True)

X_trainvalid = (
    db_mortality.drop(columns="conc1_mean")
    .iloc[trainvalid_idx, :]
    .reset_index(drop=True)
)
X_valid = (
    db_mortality.drop(columns="conc1_mean").iloc[valid_idx, :].reset_index(drop=True)
)
Y_trainvalid = db_mortality.iloc[trainvalid_idx, :].conc1_mean
Y_valid = db_mortality.iloc[valid_idx, :].conc1_mean

if "conc1_mean" in list(X_trainvalid.columns):
    print("yes")
else:
    print("no")


print("calcultaing distance matrix..", ctime())
matrix_h, matrix_p = cal_matrixs(
    X_trainvalid, X_trainvalid, categorical, non_categorical
)
print("distance matrix calculation finished", ctime())


hyper_params_tune = {
    "max_depth": [i for i in range(10, 30, 6)],
    "n_estimators": [int(x) for x in np.linspace(start=200, stop=1000, num=11)],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4, 8, 16, 32],
}

params_comb = list(ParameterSampler(hyper_params_tune, n_iter=100, random_state=rand))

best_accs = 0
best_p = dict()

if args.alpha_h == "logspace":
    sequence_ap = np.logspace(-2, 0, 20)
    sequence_ah = sequence_ap
else:
    sequence_ap = [float(args.alpha_p)]
    sequence_ah = [float(args.alpha_h)]
count = 1
for ah in sequence_ah:
    for ap in sequence_ap:
        for i in range(0, len(params_comb)):
            print(
                "*" * 50,
                count / (len(sequence_ap) ** 2 * len(params_comb)),
                ctime(),
                end="\r",
            )
            count = count + 1
            model = RandomForestClassifier()
            # model = LogisticRegression(random_state=10, n_jobs=60)
            for k, v in params_comb[i].items():
                setattr(model, k, v)

            best_results = RASAR_simple(
                df_fishchem_tv,
                col_groups,
                matrix_h,
                matrix_p,
                ah,
                ap,
                X_trainvalid,
                Y_trainvalid,
                db_invitro_matrix="noinvitro",
                n_neighbors=args.n_neighbors,
                invitro=args.w_invitro,
                invitro_form=args.invitro_label,
                db_invitro=args.db_invitro,
                encoding=encoding,
                model=model,
            )

            if best_results["avg_accs"] > best_accs:
                best_p = params_comb[i]
                best_accs = best_results["avg_accs"]
                best_results = best_results
                best_ah = ah
                best_ap = ap

# -------------------tested on test dataset--------------------
for k, v in best_p.items():
    setattr(model, k, v)


minmax = MinMaxScaler().fit(X_trainvalid[non_categorical])
X_trainvalid[non_categorical] = minmax.transform(X_trainvalid.loc[:, non_categorical])
X_valid[non_categorical] = minmax.transform(X_valid.loc[:, non_categorical])

matrix_test = dist_matrix(
    X_valid, X_trainvalid, non_categorical, categorical, best_ah, best_ap
)
matrix_train = dist_matrix(
    X_trainvalid, X_trainvalid, non_categorical, categorical, best_ah, best_ap
)

train_rf, test_rf = cal_s_rasar(
    matrix_train, matrix_test, Y_trainvalid, args.n_neighbors, encoding
)
invitro_form = args.invitro_label
db_invitro = args.db_invitro
invitro = args.w_invitro


if invitro == "own":
    train_rf = pd.DataFrame()
    test_rf = pd.DataFrame()

if str(db_invitro) == "overlap":
    if (invitro != "False") & (invitro_form == "number"):
        train_rf["invitro_conc"] = X_trainvalid.invitro_conc.reset_index(drop=True)
        test_rf["invitro_conc"] = X_valid.invitro_conc.reset_index(drop=True)
    elif (invitro != "False") & (invitro_form == "label"):
        train_rf["invitro_label"] = X_trainvalid.invitro_label.reset_index(drop=True)
        test_rf["invitro_label"] = X_valid.invitro_label.reset_index(drop=True)

    elif (invitro != "False") & (invitro_form == "both"):
        train_rf["ec50"] = X_trainvalid.invitro_conc.reset_index(drop=True)
        test_rf["ec50"] = X_valid.invitro_conc.reset_index(drop=True)
        train_rf["invitro_label"] = X_trainvalid.invitro_label.reset_index(drop=True)
        test_rf["invitro_label"] = X_valid.invitro_label.reset_index(drop=True)
else:
    train_index = X_trainvalid.index
    test_index = X_valid.index
    if (invitro != "False") & (invitro_form == "number"):
        dist = np.array(db_invitro_matrix.iloc[train_index, :].min(axis=1))
        ls = np.array(db_invitro_matrix.iloc[train_index, :].idxmin(axis=1))
        conc = db_invitro.iloc[ls, :].invitro_conc.reset_index(drop=True)
        dist = db_invitro_matrix.lookup(pd.Series(ls).index, pd.Series(ls).values)
        train_rf["invitro_conc"] = np.array(conc)
        train_rf["invitro_dist"] = dist

        dist = np.array(db_invitro_matrix.iloc[test_index, :].min(axis=1))
        ls = np.array(db_invitro_matrix.iloc[test_index, :].idxmin(axis=1))
        conc = db_invitro.iloc[ls, :].invitro_conc.reset_index(drop=True)
        dist = db_invitro_matrix.lookup(pd.Series(ls).index, pd.Series(ls).values)
        test_rf["invitro_conc"] = np.array(conc)
        test_rf["invitro_dist"] = dist

    elif (invitro != "False") & (invitro_form == "label"):
        dist = np.array(db_invitro_matrix.iloc[train_index, :].min(axis=1))
        ls = np.array(db_invitro_matrix.iloc[train_index, :].idxmin(axis=1))
        label = db_invitro.iloc[ls, :].invitro_label.reset_index(drop=True)
        dist = db_invitro_matrix.lookup(pd.Series(ls).index, pd.Series(ls).values)
        train_rf["invitro_label"] = np.array(label)
        train_rf["invitro_dist"] = dist

        dist = np.array(db_invitro_matrix.iloc[test_index, :].min(axis=1))
        ls = np.array(db_invitro_matrix.iloc[test_index, :].idxmin(axis=1))
        label = db_invitro.iloc[ls, :].invitro_label.reset_index(drop=True)
        dist = db_invitro_matrix.lookup(pd.Series(ls).index, pd.Series(ls).values)
        test_rf["invitro_label"] = np.array(label)
        test_rf["invitro_dist"] = dist

    elif (invitro != "False") & (invitro_form == "both"):

        dist = np.array(db_invitro_matrix.iloc[train_index, :].min(axis=1))
        ls = np.array(db_invitro_matrix.iloc[train_index, :].idxmin(axis=1))
        conc = db_invitro.iloc[ls, :].invitro_conc.reset_index(drop=True)
        label = db_invitro.iloc[ls, :].invitro_label.reset_index(drop=True)
        train_rf["invitro_conc"] = np.array(conc)
        train_rf["invitro_label"] = np.array(label)
        train_rf["invitro_dist"] = dist

        dist = np.array(db_invitro_matrix.iloc[test_index, :].min(axis=1))
        ls = np.array(db_invitro_matrix.iloc[test_index, :].idxmin(axis=1))
        conc = db_invitro.iloc[ls, :].invitro_conc.reset_index(drop=True)
        label = db_invitro.iloc[ls, :].invitro_label.reset_index(drop=True)
        test_rf["invitro_conc"] = np.array(conc)
        test_rf["invitro_label"] = np.array(label)
        test_rf["invitro_dist"] = dist


model.fit(train_rf, Y_trainvalid)
y_pred = model.predict(test_rf)
print(train_rf.columns)
if encoding == "binary":

    accs = accuracy_score(Y_valid, y_pred)
    sens = recall_score(Y_valid, y_pred)
    tn, fp, fn, tp = confusion_matrix(Y_valid, y_pred, labels=[0, 1]).ravel()
    specs = tn / (tn + fp)
    precs = precision_score(Y_valid, y_pred)
    f1 = f1_score(Y_valid, y_pred)

elif encoding == "multiclass":
    accs = accuracy_score(Y_valid, y_pred)
    sens = recall_score(Y_valid, y_pred, average="macro")
    specs = np.nan
    precs = precision_score(Y_valid, y_pred, average="macro")
    f1 = f1_score(Y_valid, y_pred, average="macro")


print(
    """Accuracy:  {}, Se.Accuracy:  {} 
		\nSensitivity:  {}, Se.Sensitivity: {}
        \nSpecificity:  {}, Se.Specificity:{}
		\nPrecision:  {}, Se.Precision: {}
		\nf1_score:{}, Se.f1_score:{}""".format(
        accs,
        best_results["se_accs"],
        sens,
        best_results["se_sens"],
        specs,
        best_results["se_specs"],
        precs,
        best_results["se_precs"],
        f1,
        best_results["se_f1"],
    )
)

info = []
info.append(
    """Accuracy:  {}, Se.Accuracy:  {} 
		\nSensitivity:  {}, Se.Sensitivity: {}
        \nSpecificity:  {}, Se.Specificity:{}
		\nPrecision:  {}, Se.Precision: {}
		\nf1_score:{}, Se.f1_score:{}""".format(
        accs,
        best_results["se_accs"],
        sens,
        best_results["se_sens"],
        specs,
        best_results["se_specs"],
        precs,
        best_results["se_precs"],
        f1,
        best_results["se_f1"],
    )
)
info.append("Alpha_h:{}, Alpha_p: {}".format(args.alpha_h, args.alpha_p))
info.append("Random state:{}".format(rand))

filename = args.outputFile
dirname = os.path.dirname(filename)
if not os.path.exists(dirname):
    os.makedirs(dirname)

with open(filename, "w") as file_handler:
    for item in info:
        file_handler.write("{}\n".format(item))


# python RASAR_simple_cte.py -i1 lc_db_processed.csv -ah 0.0749 -ap 0.5623  -o rasar/s_rf_bi.txt
# python RASAR_simple_cte.py -i1 lc_db_processed.csv -ah 0.31622776601683794 -ap 3.625  -o rasar/s_rf_bi2.txt
# python RASAR_simple_cte.py -i1 lc_db_processed.csv -ah 1 -ap 1  -o rasar/s_lr_tab4_bi.txt

# python RASAR_simple_cte.py -i1 lc_db_processed.csv -e "multiclass" -ah 0.01268961003167922 -ap 0.3562247890262442  -o rasar/s_rf_mul_test.txt
# python RASAR_df.py -i1 lc_db_processed.csv  -idf datafusion_db_processed.csv -e "multiclass" -label ['LC50','EC50'] -effect 'MOR' -fixed no -ah 0.01268961003167922 -ap 0.3562247890262442  -o rasar/df_mul_test.txt
#

# python RASAR_simple_cte.py -i1 lc_db_processed.csv -ah 0.0749 -ap 0.5623  -o rasar/s_rf_bi.txt
# python RASAR_df.py -i1 lc_db_processed.csv  -idf datafusion_db_processed.csv  -label ['LC50','EC50'] -effect 'MOR' -fixed no -ah 0.0749 -ap 0.5623 -o rasar/df_bi.txt

# python RASAR_simple_cte.py -i1 lc_db_processed.csv -e "multiclass" -ah 0.01268961003167922 -ap 0.3562247890262442  -o rasar/s_rf_mul.txt
# python RASAR_df.py -i1 lc_db_processed.csv  -idf datafusion_db_processed.csv -e "multiclass" -label ['LC50','EC50'] -effect 'MOR' -fixed no -ah 0.01268961003167922 -ap 0.3562247890262442  -o rasar/df_mul.txt
#


# python RASAR_simple_cte.py -i1 data/invivo/invivo_repeated_w_invitro.csv -n 1 -ah 0.01 -ap 0.15264179671752334 -dbi "overlap" -wi "True" -il "number" -o invitro/repeat_number.txt
# python RASAR_simple_cte.py -i1 data/invivo/invivo_repeated_w_invitro.csv -n 1 -ah 0.01 -ap 0.15264179671752334 -dbi "overlap" -wi "True" -il "label" -o invitro/repeat_label_half.txt
# python RASAR_simple_cte.py -i1 data/invivo/invivo_repeated_w_invitro.csv -n 1 -ah 0.01 -ap 0.15264179671752334 -dbi "overlap" -wi "True" -il "both" -o invitro/repeat_both_half.txt
# python RASAR_simple_cte.py -i1 data/invivo/invivo_repeated_w_invitro.csv -n 1 -ah 0.01 -ap 0.15264179671752334 -dbi "no" -wi "False" -il "both" -o invitro/repeat.txt

# python RASAR_simple_cte.py -i1 data/invivo/invivo_repeated_w_invitro.csv -n 1 -ah 0.01 -ap 0.15264179671752334 -dbi "overlap" -wi "own" -il "number" -o invitro/repeat_own_number.txt
# python RASAR_simple_cte.py -i1 data/invivo/invivo_repeated_w_invitro.csv -n 1 -ah 0.01 -ap 0.15264179671752334 -dbi "overlap" -wi "own" -il "label" -o invitro/repeat_own_label.txt
# python RASAR_simple_cte.py -i1 data/invivo/invivo_repeated_w_invitro.csv -n 1 -ah 0.01 -ap 0.15264179671752334 -dbi "overlap" -wi "own" -il "both" -o invitro/repeat_own_both.txt
#


# python RASAR_simple_cte.py -i1 data/invivo/invivo_repeated_w_invitro.csv -e "multiclass" -n 1 -ah 0.01 -ap 0.15264179671752334 -dbi "overlap" -wi "True" -il "number" -o invitro/repeat_number.txt
# python RASAR_simple_cte.py -i1 data/invivo/invivo_repeated_w_invitro.csv -e "multiclass" -n 1 -ah 0.01 -ap 0.15264179671752334 -dbi "overlap" -wi "True" -il "label" -o invitro/repeat_label_half.txt
# python RASAR_simple_cte.py -i1 data/invivo/invivo_repeated_w_invitro.csv -e "multiclass" -n 1 -ah 0.01 -ap 0.15264179671752334 -dbi "overlap" -wi "True" -il "both" -o invitro/repeat_both_half.txt
# python RASAR_simple_cte.py -i1 data/invivo/invivo_repeated_w_invitro.csv -e "multiclass" -n 1 -ah 0.01 -ap 0.15264179671752334 -dbi "no" -wi "False" -il "both" -o invitro/repeat.txt

# python RASAR_simple_cte.py -i1 data/invivo/invivo_repeated_w_invitro.csv -e "multiclass" -n 1 -ah 0.01 -ap 0.15264179671752334 -dbi "overlap" -wi "own" -il "number" -o invitro/repeat_own_number.txt
# python RASAR_simple_cte.py -i1 data/invivo/invivo_repeated_w_invitro.csv -e "multiclass" -n 1 -ah 0.01 -ap 0.15264179671752334 -dbi "overlap" -wi "own" -il "label" -o invitro/repeat_own_label.txt
# python RASAR_simple_cte.py -i1 data/invivo/invivo_repeated_w_invitro.csv -e "multiclass" -n 1 -ah 0.01 -ap 0.15264179671752334 -dbi "overlap" -wi "own" -il "both" -o invitro/repeat_own_both.txt
#


# ---------------------best R find----------------------
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 1 -ah 0.02592943797404667 -ap 10.0  -o bestR/s_rasar_3nn_1.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 2 -ah 0.02592943797404667 -ap 10.0  -o bestR/s_rasar_3nn_2.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 3 -ah 0.02592943797404667 -ap 10.0  -o bestR/s_rasar_3nn_3.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 4 -ah 0.02592943797404667 -ap 10.0  -o bestR/s_rasar_3nn_4.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 5 -ah 0.02592943797404667 -ap 10.0  -o bestR/s_rasar_3nn_5.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 7 -ah 0.02592943797404667 -ap 10.0  -o bestR/s_rasar_3nn_7.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 9 -ah 0.02592943797404667 -ap 10.0  -o bestR/s_rasar_3nn_9.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 11 -ah 0.02592943797404667 -ap 10.0  -o bestR/s_rasar_3nn_11.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 13 -ah 0.02592943797404667 -ap 10.0  -o bestR/s_rasar_3nn_13.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 15 -ah 0.02592943797404667 -ap 10.0  -o bestR/s_rasar_3nn_15.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 1  -ah 0.02592943797404667 -ap 10.0  -o bestR/s_rasar_5nn_1.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 2  -ah 0.02592943797404667 -ap 10.0  -o bestR/s_rasar_5nn_2.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 3  -ah 0.02592943797404667 -ap 10.0  -o bestR/s_rasar_5nn_3.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 4  -ah 0.02592943797404667 -ap 10.0  -o bestR/s_rasar_5nn_4.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 5  -ah 0.02592943797404667 -ap 10.0  -o bestR/s_rasar_5nn_5.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 7  -ah 0.02592943797404667 -ap 10.0  -o bestR/s_rasar_5nn_7.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 9  -ah 0.02592943797404667 -ap 10.0  -o bestR/s_rasar_5nn_9.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 11 -ah 0.02592943797404667 -ap 10.0  -o bestR/s_rasar_5nn_11.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 13 -ah 0.02592943797404667 -ap 10.0  -o bestR/s_rasar_5nn_13.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 15 -ah 0.02592943797404667 -ap 10.0  -o bestR/s_rasar_5nn_15.txt

# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 1  -ah 1 -ap 1 -o bestR/s_rasar_euc_1.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 2  -ah 1 -ap 1 -o bestR/s_rasar_euc_2.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 3  -ah 1 -ap 1 -o bestR/s_rasar_euc_3.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 4  -ah 1 -ap 1 -o bestR/s_rasar_euc_4.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 5  -ah 1 -ap 1 -o bestR/s_rasar_euc_5.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 7  -ah 1 -ap 1 -o bestR/s_rasar_euc_7.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 9  -ah 1 -ap 1 -o bestR/s_rasar_euc_9.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 11 -ah 1 -ap 1 -o bestR/s_rasar_euc_11.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 13 -ah 1 -ap 1 -o bestR/s_rasar_euc_13.txt
# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 15 -ah 1 -ap 1 -o bestR/s_rasar_euc_15.txt

# python RASAR_simple_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv -n 1  -e "multiclass" -ah 1 -ap 1 -o bestR/s_rasar_euc_mul_1.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 2  -e "multiclass" -ah 1 -ap 1 -o bestR/s_rasar_euc_mul_2.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 3  -e "multiclass" -ah 1 -ap 1 -o bestR/s_rasar_euc_mul_3.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 4  -e "multiclass" -ah 1 -ap 1 -o bestR/s_rasar_euc_mul_4.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 5  -e "multiclass" -ah 1 -ap 1 -o bestR/s_rasar_euc_mul_5.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 7  -e "multiclass" -ah 1 -ap 1 -o bestR/s_rasar_euc_mul_7.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 9  -e "multiclass" -ah 1 -ap 1 -o bestR/s_rasar_euc_mul_9.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 11 -e "multiclass" -ah 1 -ap 1 -o bestR/s_rasar_euc_mul_11.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 13 -e "multiclass" -ah 1 -ap 1 -o bestR/s_rasar_euc_mul_13.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 15 -e "multiclass" -ah 1 -ap 1 -o bestR/s_rasar_euc_mul_15.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 1  -e "multiclass" -ah 0.01 -ap 4.832930238571752 -o rasar/s_rasar_3nn_mul_1.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 2  -e "multiclass" -ah 0.01 -ap 4.832930238571752 -o bestR/s_rasar_3nn_mul_2.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 3  -e "multiclass" -ah 0.01 -ap 4.832930238571752 -o bestR/s_rasar_3nn_mul_3.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 4  -e "multiclass" -ah 0.01 -ap 4.832930238571752 -o bestR/s_rasar_3nn_mul_4.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 5  -e "multiclass" -ah 0.01 -ap 4.832930238571752 -o bestR/s_rasar_3nn_mul_5.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 7  -e "multiclass" -ah 0.01 -ap 4.832930238571752 -o bestR/s_rasar_3nn_mul_7.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 9  -e "multiclass" -ah 0.01 -ap 4.832930238571752 -o bestR/s_rasar_3nn_mul_9.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 11 -e "multiclass" -ah 0.01 -ap 4.832930238571752 -o bestR/s_rasar_3nn_mul_11.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 13 -e "multiclass" -ah 0.01 -ap 4.832930238571752 -o bestR/s_rasar_3nn_mul_13.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 15 -e "multiclass" -ah 0.01 -ap 4.832930238571752 -o bestR/s_rasar_3nn_mul_15.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 1  -e "multiclass" -ah 0.03290344562312668 -ap 3.856620421163472 -o bestR/s_rasar_5nn_mul_1.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 2  -e "multiclass" -ah 0.03290344562312668 -ap 3.856620421163472 -o bestR/s_rasar_5nn_mul_2.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 3  -e "multiclass" -ah 0.03290344562312668 -ap 3.856620421163472 -o bestR/s_rasar_5nn_mul_3.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 4  -e "multiclass" -ah 0.03290344562312668 -ap 3.856620421163472 -o bestR/s_rasar_5nn_mul_4.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 5  -e "multiclass" -ah 0.03290344562312668 -ap 3.856620421163472 -o bestR/s_rasar_5nn_mul_5.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 7  -e "multiclass" -ah 0.03290344562312668 -ap 3.856620421163472 -o bestR/s_rasar_5nn_mul_7.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 9  -e "multiclass" -ah 0.03290344562312668 -ap 3.856620421163472 -o bestR/s_rasar_5nn_mul_9.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 11 -e "multiclass" -ah 0.03290344562312668 -ap 3.856620421163472 -o bestR/s_rasar_5nn_mul_11.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 13 -e "multiclass" -ah 0.03290344562312668 -ap 3.856620421163472 -o bestR/s_rasar_5nn_mul_13.txt
# python RASAR_simple_cte.py -i1 data/invivo/lc50_processed_jim.csv -n 15 -e "multiclass" -ah 0.03290344562312668 -ap 3.856620421163472 -o bestR/s_rasar_5nn_mul_15.txt
