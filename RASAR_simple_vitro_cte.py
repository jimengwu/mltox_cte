from helper_cte_model import *
from sklearn.model_selection import ParameterSampler
import argparse


def getArguments():
    parser = argparse.ArgumentParser(
        description="Simple rasar model with adding invitro."
    )
    parser.add_argument("-i", "--input", dest="inputFile", required=True)
    parser.add_argument("-iv", "--input_vitro", dest="inputFile_vitro", required=True)
    parser.add_argument(
        "-il", "--invitro_label", dest="invitro_label", default="number"
    )
    parser.add_argument("-dbi", "--db_invitro", dest="db_invitro", default="no")
    parser.add_argument("-wi", "--w_invitro", dest="w_invitro", default="False")
    parser.add_argument("-ni", "--niter", dest="niter", default=50, type=int)
    parser.add_argument(
        "-n", "--n_neighbors", dest="n_neighbors", nargs="?", default=1, type=int
    )
    parser.add_argument("-e", "--encoding", dest="encoding", default="binary")
    parser.add_argument(
        "-m", "--model", help="model: logistic regression, random forest", default="rf",
    )
    parser.add_argument("-ah", "--alpha_h", dest="alpha_h", required=True, nargs="?")
    parser.add_argument("-ap", "--alpha_p", dest="alpha_p", required=True, nargs="?")
    parser.add_argument("-o", "--output", dest="outputFile", default="binary.txt")
    return parser.parse_args()


args = getArguments()

if args.encoding == "binary":
    encoding = "binary"
    encoding_value = 1
elif args.encoding == "multiclass":
    encoding = "multiclass"
    encoding_value = [0.1, 1, 10, 100]

# -----------loading data & splitting into train and test dataset--------
print("loading dataset...", ctime())
X, Y, db_invitro = load_invivo_invitro(
    args.inputFile,
    args.inputFile_vitro,
    encoding=encoding,
    encoding_value=encoding_value,
    seed=42,
)
print("finish loaded.", ctime())
test_size = 0.2
col_groups = "test_cas"

df_fishchem = X[["fish", "test_cas"]]
traintest_idx, valid_idx = get_grouped_train_test_split(
    df_fishchem, test_size, col_groups
)
df_fishchem_tv = df_fishchem.iloc[traintest_idx, :]

# Encoding for invitro variable: binary and multiclass
if args.encoding == "binary":
    db_invitro["invitro_label"] = np.where(db_invitro["invitro_conc"].values > 1, 0, 1)
    db_invitro["invitro_label_half"] = np.where(
        db_invitro["invitro_conc"].median() > 1, 0, 1
    )
elif args.encoding == "multiclass":
    db_invitro["invitro_label"] = multiclass_encoding(
        db_invitro["invitro_conc"], [0.006, 0.3, 63, 398]
    )


X_traintest = X.iloc[traintest_idx, :]
X_valid = X.iloc[valid_idx, :]
Y_traintest = Y[traintest_idx]
Y_valid = Y[valid_idx]

# -----------------creating the distance matrix--------

print("calcultaing distance matrix..", ctime())
matrices = cal_matrixs(X_traintest, X_traintest, categorical, non_categorical)
matrices_invitro = cal_matrixs(
    X_traintest, db_invitro, categorical_both, non_categorical
)
print("distance matrix calculation finished.", ctime())

# ------------------------hyperparameters range---------
if args.alpha_h == "logspace":
    sequence_ap = np.logspace(-2, 0, 2)
    sequence_ah = sequence_ap
else:
    sequence_ap = [float(args.alpha_p)]
    sequence_ah = [float(args.alpha_h)]


hyper_params_tune = {
    # "max_depth": [i for i in range(10, 20, 2)],
    # "n_estimators": [int(x) for x in np.linspace(start=100, stop=1000, num=2)],
    # "min_samples_split": [2, 5, 10],
    # "min_samples_leaf": [1, 2, 4],
    "n_estimators": [int(x) for x in np.linspace(start=1, stop=5, num=5)],
    "max_features": [i for i in range(1, 4, 1)],
    "max_depth": [i for i in np.linspace(start=0.01, stop=1, num=5)],
    #     "criterion":["gini", "entropy"],
    "min_samples_leaf": [i for i in np.linspace(start=0.01, stop=0.5, num=5)],
    "min_samples_split": [i for i in np.linspace(start=0.1, stop=1, num=5)],
}

params_comb = list(
    ParameterSampler(hyper_params_tune, n_iter=args.niter, random_state=2)
)

# -------------------training using the default model setting and found the best alphas combination--------------------

best_accs = 0
best_param = dict()

count = 1
print("training process start..", ctime())

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

            result = RASAR_simple(
                df_fishchem_tv,
                col_groups,
                matrices["euc"],
                matrices["hamming"],
                matrices["pubchem"],
                ah,
                ap,
                X_traintest,
                Y_traintest,
                db_invitro_matrices=matrices_invitro,
                invitro=args.w_invitro,
                n_neighbors=args.n_neighbors,
                invitro_form=args.invitro_label,
                db_invitro=db_invitro,
                encoding=args.encoding,
                model=model,
            )
            if result.accuracy.mean() > best_accs:
                best_param = params_comb[i]
                best_accs = result.accuracy.mean()
                best_result = result


df_mean = pd.DataFrame(best_result.mean(axis=0)).transpose()
df_std = pd.DataFrame(best_result.sem(axis=0)).transpose()
df_std["neighbors"] = args.n_neighbors

# -------------------tested on test dataset--------------------
print("testing start.", ctime())
for k, v in best_param.items():
    setattr(model, k, v)


minmax = MinMaxScaler().fit(X_traintest[non_categorical])
X_traintest[non_categorical] = minmax.transform(X_traintest.loc[:, non_categorical])
X_valid[non_categorical] = minmax.transform(X_valid.loc[:, non_categorical])
X[non_categorical] = minmax.transform(X.loc[:, non_categorical])
db_invitro[non_categorical] = minmax.transform(db_invitro.loc[:, non_categorical])

matrix_valid = dist_matrix(
    X_valid,
    X_traintest,
    non_categorical,
    categorical,
    best_result.iloc[0]["ah"],
    best_result.iloc[0]["ap"],
)
matrix_traintest = dist_matrix(
    X_traintest,
    X_traintest,
    non_categorical,
    categorical,
    best_result.iloc[0]["ah"],
    best_result.iloc[0]["ap"],
)


db_invitro_matrix = dist_matrix(
    X,
    db_invitro,
    non_categorical,
    categorical_both,
    best_result.iloc[0]["ah"],
    best_result.iloc[0]["ap"],
)

train_index = X_traintest.index
test_index = X_valid.index
train_rf, test_rf = cal_s_rasar(
    matrix_traintest, matrix_valid, Y_traintest, args.n_neighbors, args.encoding
)


if args.w_invitro == "own":
    train_rf = pd.DataFrame()
    test_rf = pd.DataFrame()

if args.w_invitro != "False":
    if str(args.db_invitro) == "overlap":
        train_rf = get_vitroinfo(train_rf, X, train_index, args.invitro_label)
        test_rf = get_vitroinfo(test_rf, X, test_index, args.invitro_label)
    else:
        train_rf = find_nearest_vitro(
            train_rf, db_invitro, db_invitro_matrix, train_index, args.invitro_label,
        )
        test_rf = find_nearest_vitro(
            test_rf, db_invitro, db_invitro_matrix, test_index, args.invitro_label,
        )

df_test_score = fit_and_predict(
    model, train_rf, Y_traintest, test_rf, Y_valid, args.encoding
)
for k, v in best_param.items():
    df_test_score.loc[0, k] = v

df_test_score.loc[0, ["ah", "ap"]] = best_result.iloc[0][["ah", "ap"]]
df_test_score.loc[0, "encoding"] = args.encoding

df_output = pd.concat(
    [df_mean, df_std, df_test_score],
    keys=["train_mean", "train_std", "test"],
    names=["series_name"],
)


print(df_output)
# ----------------save the information into a file-------
df2file(df_output, args.outputFile + ".txt")


# ----------------------------------------------------------------------------------general: invitro + (invivo) -> invivo (gtv/gtvv)--------------------------------------------
# binary, R=4
# python RASAR_simple_addinginvitro_general_cte.py -i /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -iv /local/wujimeng/code_jimeng/data/invitro/invitro_processed.csv  -wi "True" -il "number" -n 4 -ah 0.23357214690901212 -ap 0.1128837891684889  -o "vitro_e/vivo+vitro/bestR/general/general.txt"
# python RASAR_simple_addinginvitro_general_cte.py -i /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -iv /local/wujimeng/code_jimeng/data/invitro/invitro_processed.csv  -wi "own"  -il "number" -n 4 -ah 0.23357214690901212 -ap 0.1128837891684889  -o "vitro_e/vivo+vitro/bestR/general/general_own_invitro.txt"

# multiclass, R=5
# python RASAR_simple_addinginvitro_general_cte.py -i data/invivo/lc50_processed_jim.csv  -i2 data/invitro/invitro_processed.csv -e "multiclass" -wi "True" -il "number" -n 5 -ah 0.04281332398719394 -ap 0.7847599703514611  -o "vitro_e/vivo+vitro/bestR/multiclass/general/general_mul.txt"
# python RASAR_simple_addinginvitro_general_cte.py -i data/invivo/lc50_processed_jim.csv  -i2 data/invitro/invitro_processed.csv -e "multiclass" -wi "own"  -il "number" -n 5 -ah 0.04281332398719394 -ap 0.7847599703514611  -o "vitro_e/vivo+vitro/bestR/multiclass/general/general_own_invitro_mul.txt"


# model.fit(train_rf, Y_traintest)
# y_pred = model.predict(test_rf)


# if args.encoding == "binary":

#     accs = accuracy_score(Y_test, y_pred)
#     sens = recall_score(Y_test, y_pred, average="macro")
#     tn, fp, fn, tp = confusion_matrix(Y_test, y_pred, labels=[0, 1]).ravel()
#     specs = tn / (tn + fp)
#     precs = precision_score(Y_test, y_pred, average="macro")
#     f1 = f1_score(Y_test, y_pred, average="macro")

# elif args.encoding == "multiclass":
#     accs = accuracy_score(Y_test, y_pred)
#     sens = recall_score(Y_test, y_pred, average="macro")
#     specs = np.nan
#     precs = precision_score(Y_test, y_pred, average="macro")
#     f1 = f1_score(Y_test, y_pred, average="macro")


# print(
#     """Accuracy:  {}, Se.Accuracy:  {}
#         \nSensitivity:  {}, Se.Sensitivity: {}
#         \nSpecificity:  {}, Se.Specificity:{}
#         \nPrecision:  {}, Se.Precision: {}
#         \nf1_score:{}, Se.f1_score:{}""".format(
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
#         \nSensitivity:  {}, Se.Sensitivity: {}
#         \nSpecificity:  {}, Se.Specificity:{}
#         \nPrecision:  {}, Se.Precision: {}
#         \nf1_score:{}, Se.f1_score:{}""".format(
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
# info.append("Alpha_h:{}, Alpha_p: {},n:{}".format(best_ah, best_ap, args.n_neighbors))

# filename = args.outputFile
# dirname = os.path.dirname(filename)
# if not os.path.exists(dirname):
#     os.makedirs(dirname)

# with open(filename, "w") as file_handler:
#     for item in info:
#         file_handler.write("{}\n".format(item))


# ------------------------------------------------------old version
# waiting:


# python RASAR_simple_addinginvitro_general2.py -i data/invivo/lc50_processed_jim.csv  -i2 data/invitro/invitro_merged.csv  -wi "True" -il "number" -n 1 -ah "logspace" -ap "logspace"  -o invitro/general_alphas.txt
# python RASAR_simple_addinginvitro_general2.py -i data/invivo/lc50_processed_jim.csv  -i2 data/invitro/invitro_merged.csv  -wi "own"  -il "number" -n 1 -ah "logspace" -ap "logspace"  -o invitro/general_own_alphas.txt


# python RASAR_simple_addinginvitro_general2.py -i data/invivo/lc50_processed_jim.csv  -i2 data/invitro/invitro_merged.csv  -wi "True" -il "number" -n 1 -ah 0.041753189365604 -ap 4.893900918477494  -o "vivo+vitro(t&e)/general.txt"
# python RASAR_simple_addinginvitro_general2.py -i data/invivo/lc50_processed_jim.csv  -i2 data/invitro/invitro_merged.csv  -wi "own"  -il "number" -n 1 -ah 0.041753189365604 -ap 4.893900918477494  -o "vivo+vitro(t&e)/general_own_invitro.txt"

# python RASAR_simple_addinginvitro_general2.py -i data/invivo/lc50_processed_jim.csv  -i2 data/invitro/invitro_merged.csv  -wi "False" -il "number" -n 1 -ah 0.041753189365604 -ap 4.893900918477494  -o vivo+vitro(t&e)/general_compared.txt


# python RASAR_simple_addinginvitro_general2.py -i data/invivo/lc50_processed_jim.csv  -i2 data/invitro/invitro_merged.csv     -wi "True" -il "number" -n 4 -ah 0.23357214690901212 -ap 0.1128837891684889  -o "vivo+vitro(t&e)/bestR/general.txt"
# python RASAR_simple_addinginvitro_general2.py -i data/invivo/lc50_processed_jim.csv  -i2 data/invitro/invitro_merged.csv     -wi "own"  -il "number" -n 4 -ah 0.23357214690901212 -ap 0.1128837891684889  -o "vivo+vitro(t&e)/bestR/general_own_invitro.txt"
# python RASAR_simple_addinginvitro_general2.py -i data/invivo/lc50_processed_jim.csv  -i2 data/TOXCAST/toxcast_processed.csv  -wi "True" -il "number" -n 4 -ah 0.23357214690901212 -ap 0.1128837891684889  -o "vitro_t/vivo&vitro/bestR/general.txt"
# python RASAR_simple_addinginvitro_general2.py -i data/invivo/lc50_processed_jim.csv  -i2 data/TOXCAST/toxcast_processed.csv  -wi "own"  -il "number" -n 4 -ah 0.23357214690901212 -ap 0.1128837891684889  -o "vitro_t/vivo&vitro/bestR/general_own_invitro.txt"
# python RASAR_simple_addinginvitro_general2.py -i data/invivo/lc50_processed_jim.csv  -i2 data/invitro/invitro_merged.csv     -wi "True" -il "number" -n 1 -ah 0.041753189365604   -ap 4.893900918477494   -o "vivo+vitro(t&e)/general.txt"
# python RASAR_simple_addinginvitro_general2.py -i data/invivo/lc50_processed_jim.csv  -i2 data/invitro/invitro_merged.csv     -wi "own"  -il "number" -n 1 -ah 0.041753189365604   -ap 4.893900918477494   -o "vivo+vitro(t&e)/general_own_invitro.txt"
#


# invitro_form = args.invitro_label
# invitro = args.w_invitro

# if invitro == "own":
#     train_rf = pd.DataFrame()
#     test_rf = pd.DataFrame()

# if str(db_invitro) == "overlap":
#     if (invitro != "False") & (invitro_form == "number"):
#         train_rf["invitro_conc"] = X_train.invitro_conc.reset_index(drop=True)
#         test_rf["invitro_conc"] = X_valid.invitro_conc.reset_index(drop=True)

#     elif (invitro != "False") & (invitro_form == "label"):
#         train_rf["invitro_label"] = X_train.invitro_label.reset_index(drop=True)
#         test_rf["invitro_label"] = X_valid.invitro_label.reset_index(drop=True)

#     elif (invitro != "False") & (invitro_form == "both"):
#         train_rf["invitro_conc"] = X_train.invitro_conc.reset_index(drop=True)
#         test_rf["invitro_conc"] = X_valid.invitro_conc.reset_index(drop=True)
#         train_rf["invitro_label"] = X_train.invitro_label.reset_index(drop=True)
#         test_rf["invitro_label"] = X_valid.invitro_label.reset_index(drop=True)
# else:
#     if (invitro != "False") & (invitro_form == "number"):
#         ls = np.array(db_invitro_matrix_train.idxmin(axis=1))
#         # dist = db_invitro_matrix_train.lookup(pd.Series(ls).index, pd.Series(ls).values)
#         dist = np.array(db_invitro_matrix_train.min(axis=1))
#         conc = db_invitro.iloc[ls, :].invitro_conc.reset_index(drop=True)
#         train_rf["invitro_conc"] = np.array(conc)
#         train_rf["invitro_dist"] = dist

#         ls = np.array(db_invitro_matrix_test.idxmin(axis=1))
#         # dist = db_invitro_matrix_test.lookup(pd.Series(ls).index, pd.Series(ls).values)
#         dist = np.array(db_invitro_matrix_test.min(axis=1))
#         conc = db_invitro.iloc[ls, :].invitro_conc.reset_index(drop=True)
#         test_rf["invitro_conc"] = np.array(conc)
#         test_rf["invitro_dist"] = dist
#         # print(np.array(conc))

#     elif (invitro != "False") & (invitro_form == "label"):
#         dist = np.array(db_invitro_matrix_train.min(axis=1))
#         ls = np.array(db_invitro_matrix_train.idxmin(axis=1))
#         label = db_invitro.iloc[ls, :].invitro_label.reset_index(drop=True)
#         # dist = db_invitro_matrix_train.lookup(pd.Series(ls).index, pd.Series(ls).values)
#         train_rf["invitro_label"] = np.array(label)
#         train_rf["invitro_dist"] = dist

#         dist = np.array(db_invitro_matrix_test.min(axis=1))
#         ls = np.array(db_invitro_matrix_test.idxmin(axis=1))
#         label = db_invitro.iloc[ls, :].invitro_label.reset_index(drop=True)
#         # dist = db_invitro_matrix_test.lookup(pd.Series(ls).index, pd.Series(ls).values)
#         test_rf["invitro_label"] = np.array(label)
#         test_rf["invitro_dist"] = dist

#     elif (invitro != "False") & (invitro_form == "both"):

#         dist = np.array(db_invitro_matrix_train.min(axis=1))
#         ls = np.array(db_invitro_matrix_train.idxmin(axis=1))
#         conc = db_invitro.iloc[ls, :].invitro_conc.reset_index(drop=True)
#         label = db_invitro.iloc[ls, :].invitro_label.reset_index(drop=True)
#         train_rf["invitro_conc"] = np.array(conc)
#         train_rf["invitro_label"] = np.array(label)
#         train_rf["invitro_dist"] = dist

#         dist = np.array(db_invitro_matrix_test.min(axis=1))
#         ls = np.array(db_invitro_matrix_test.idxmin(axis=1))
#         conc = db_invitro.iloc[ls, :].invitro_conc.reset_index(drop=True)
#         label = db_invitro.iloc[ls, :].invitro_label.reset_index(drop=True)
#         test_rf["invitro_conc"] = np.array(conc)
#         test_rf["invitro_label"] = np.array(label)
#         test_rf["invitro_dist"] = dist
