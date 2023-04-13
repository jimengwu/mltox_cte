from helper_cte_model import *
from sklearn.model_selection import ParameterSampler
from sklearn.ensemble import RandomForestClassifier
import argparse
import os


def getArguments():
    parser = argparse.ArgumentParser(description="Running KNN_model for datasets.")
    parser.add_argument("-i", "--input", dest="inputFile", required=True)
    parser.add_argument("-ah", "--alpha_h", dest="alpha_h", required=True, nargs="?")
    parser.add_argument("-ap", "--alpha_p", dest="alpha_p", required=True, nargs="?")
    parser.add_argument("-dbi", "--db_invitro", dest="db_invitro", default="noinvitro")
    parser.add_argument("-e", "--encoding", dest="encoding", default="binary")
    parser.add_argument(
        "-if",
        "--invitroFile",
        dest="invitroFile",
        help="if input is invitroFile",
        default=False,
    )
    parser.add_argument(
        "-il", "--invitro_label", dest="invitro_label", default="number"
    )
    parser.add_argument(
        "-m", "--model", help="model: logistic regression, random forest", default="rf",
    )
    parser.add_argument(
        "-n", "--n_neighbors", dest="n_neighbors", nargs="?", default=1, type=int
    )
    parser.add_argument(
        "-ni",
        "--niter",
        help="number of iterations to find the best parameter combination",
        nargs="?",
        default=100,
        type=int,
    )
    parser.add_argument("-wi", "--w_invitro", dest="w_invitro", default="False")
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

# -----------loading data & splitting into train and test dataset--------
db_mortality = load_data(
    args.inputFile,
    encoding=encoding,
    categorical_columns=categorical,
    conc_column=conc_column,
    encoding_value=encoding_value,
    seed=rand,
)
print("finish loaded.", ctime())

# -----------------startified split the dataset for training model--------
test_size = 0.2
col_groups = "test_cas"

df_fishchem = db_mortality[["fish", "test_cas"]]
traintest_idx, valid_idx = get_grouped_train_test_split(
    df_fishchem, test_size, col_groups
)
df_fishchem_tv = df_fishchem.iloc[traintest_idx, :].reset_index(drop=True)

X_traintest = (
    db_mortality.drop(columns="conc1_mean")
    .iloc[traintest_idx, :]
    .reset_index(drop=True)
)
X_valid = (
    db_mortality.drop(columns="conc1_mean").iloc[valid_idx, :].reset_index(drop=True)
)
Y_traintest = db_mortality.iloc[traintest_idx, :].conc1_mean
Y_valid = db_mortality.iloc[valid_idx, :].conc1_mean


# -----------------creating the distance matrix--------

print("calcultaing distance matrix..", ctime())
matrices = cal_matrixs(X_traintest, X_traintest, categorical, non_categorical)
print("distance matrix calculation finished", ctime())

# -----------------train model--------
hyper_params_tune = {
    "max_depth": [i for i in range(10, 30, 6)],
    "n_estimators": [int(x) for x in np.linspace(start=200, stop=1000, num=11)],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4, 8, 16, 32],
}

params_comb = list(
    ParameterSampler(hyper_params_tune, n_iter=args.niter, random_state=rand)
)

best_accs = 0
best_param = dict()
if args.alpha_h == "logspace":
    sequence_ap = np.logspace(-2, 0, 2)
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
            if args.model == "rf":
                model = RandomForestClassifier(random_state=10)
            elif args.model == "lr":
                model = LogisticRegression(random_state=10)

            for k, v in params_comb[i].items():
                setattr(model, k, v)

            results = RASAR_simple(
                df_fishchem_tv,
                col_groups,
                matrices["euc"],
                matrices["hamming"],
                matrices["pubchem"],
                ah,
                ap,
                X_traintest,
                Y_traintest.values,
                db_invitro_matrices="noinvitro",
                n_neighbors=args.n_neighbors,
                invitro=args.w_invitro,
                invitro_form=args.invitro_label,
                db_invitro=args.db_invitro,
                encoding=encoding,
                model=model,
            )

            if results["accuracy"].mean() > best_accs:

                best_accs = results["accuracy"].mean()

                results.loc[0, "ah"] = ah
                results.loc[0, "ap"] = ap
                for k, v in best_param.items():
                    results.loc[0, k] = v

                best_results = results

                best_param = params_comb[i]


# -------------------tested on test dataset--------------------
for k, v in best_param.items():
    setattr(model, k, v)


minmax = MinMaxScaler().fit(X_traintest[non_categorical])
X_traintest[non_categorical] = minmax.transform(X_traintest.loc[:, non_categorical])
X_valid[non_categorical] = minmax.transform(X_valid.loc[:, non_categorical])

matrix_valid = dist_matrix(
    X_valid,
    X_traintest,
    non_categorical,
    categorical,
    best_results.iloc[0]["ah"],
    best_results.iloc[0]["ap"],
)
matrix_traintest = dist_matrix(
    X_traintest,
    X_traintest,
    non_categorical,
    categorical,
    best_results.iloc[0]["ah"],
    best_results.iloc[0]["ap"],
)

traintest_rf, valid_rf = cal_s_rasar(
    matrix_traintest, matrix_valid, Y_traintest.values, args.n_neighbors, encoding
)
invitro_form = args.invitro_label
db_invitro = args.db_invitro
invitro = args.w_invitro


if invitro == "own":
    traintest_rf = pd.DataFrame()
    valid_rf = pd.DataFrame()

if invitro != "False" and str(db_invitro) == "overlap":
    traintest_rf = get_vitroinfo(traintest_rf, X_traintest, traintest_idx, invitro_form)
    valid_rf = get_vitroinfo(valid_rf, X_valid, valid_idx, invitro_form)


# --------saving the information into a file
df_output = fit_and_predict(
    model, traintest_rf, Y_traintest, valid_rf, Y_valid, encoding
)
df_output.loc[0, ["ah", "ap"]] = best_results.iloc[0][["ah", "ap"]]
for k, v in best_param.items():
    df_output.loc[0, k] = v
df_output.loc[0, "encoding"] = args.encoding

df2file(df_output, args.outputFile + ".txt")


# model.fit(traintest_rf, Y_traintest)
# y_pred = model.predict(valid_rf)
# print(traintest_rf.columns)

# if encoding == "binary":
#     accs = accuracy_score(Y_valid, y_pred)
#     sens = recall_score(Y_valid, y_pred)
#     tn, fp, fn, tp = confusion_matrix(Y_valid, y_pred, labels=[0, 1]).ravel()
#     specs = tn / (tn + fp)
#     precs = precision_score(Y_valid, y_pred)
#     f1 = f1_score(Y_valid, y_pred)

# elif encoding == "multiclass":
#     accs = accuracy_score(Y_valid, y_pred)
#     sens = recall_score(Y_valid, y_pred, average="macro")
#     specs = np.nan
#     precs = precision_score(Y_valid, y_pred, average="macro")
#     f1 = f1_score(Y_valid, y_pred, average="macro")


# print(
#     """Accuracy:  {}, Se.Accuracy:  {}
# 		\nSensitivity:  {}, Se.Sensitivity: {}
#         \nSpecificity:  {}, Se.Specificity:{}
# 		\nPrecision:  {}, Se.Precision: {}
# 		\nf1_score:{}, Se.f1_score:{}""".format(
#         accs,
#         best_results["se_accs"],
#         sens,
#         best_results["se_sens"],
#         specs,
#         best_results["se_specs"],
#         precs,
#         best_results["se_precs"],
#         f1,
#         best_results["se_f1"],
#     )
# )

# info = []
# info.append(
#     """Accuracy:  {}, Se.Accuracy:  {}
# 		\nSensitivity:  {}, Se.Sensitivity: {}
#         \nSpecificity:  {}, Se.Specificity:{}
# 		\nPrecision:  {}, Se.Precision: {}
# 		\nf1_score:{}, Se.f1_score:{}""".format(
#         accs,
#         best_results["se_accs"],
#         sens,
#         best_results["se_sens"],
#         specs,
#         best_results["se_specs"],
#         precs,
#         best_results["se_precs"],
#         f1,
#         best_results["se_f1"],
#     )
# )
# info.append("Alpha_h:{}, Alpha_p: {}".format(args.alpha_h, args.alpha_p))
# info.append("Random state:{}".format(rand))

# filename = args.outputFile
# dirname = os.path.dirname(filename)
# if not os.path.exists(dirname):
#     os.makedirs(dirname)

# with open(filename, "w") as file_handler:
#     for item in info:
#         file_handler.write("{}\n".format(item))


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
