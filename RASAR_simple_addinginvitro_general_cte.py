from helper_model_cte import *
from sklearn.model_selection import train_test_split, ParameterSampler
import argparse
import sys
import os


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
        "-m",
        "--model",
        help="model: logistic regression, random forest",
        default="rf",
    )
    parser.add_argument(
        "-ah", "--alpha_h", dest="alpha_h", default="logspace", nargs="?"
    )
    parser.add_argument("-ap", "--alpha_p", dest="alpha_p", nargs="?")
    parser.add_argument("-o", "--output", dest="outputFile", default="binary.txt")
    return parser.parse_args()


args = getArguments()

if args.encoding == "binary":
    encoding = "binary"
    encoding_value = 1
elif args.encoding == "multiclass":
    encoding = "multiclass"
    encoding_value = [0.1, 1, 10, 100]

print("loading dataset...", ctime())
X, Y, db_invitro = load_invivo_invitro(
    args.inputFile,
    args.inputFile_vitro,
    encoding=encoding,
    encoding_value=encoding_value,
    seed=42,
)
# X = X[:200]
# Y = Y[:200]
print("finish loaded.", ctime())
test_size = 0.2
col_groups = "test_cas"

df_fishchem = X[["fish", "test_cas"]]
trainvalid_idx, valid_idx = get_grouped_train_test_split(
    df_fishchem, test_size, col_groups, rand=0
)
df_fishchem_tv = df_fishchem.iloc[trainvalid_idx, :]


if args.encoding == "binary":
    db_invitro["invitro_label"] = np.where(db_invitro["invitro_conc"].values > 1, 0, 1)
elif args.encoding == "multiclass":
    db_invitro["invitro_label"] = multiclass_encoding(
        db_invitro["invitro_conc"], [0.006, 0.3, 63, 398]
    )


# X_trainvalid, X_valid, Y_trainvalid, Y_test = train_test_split(
#     X, Y, test_size=0.2, random_state=42
# )


X_trainvalid = X.iloc[trainvalid_idx, :]
X_valid = X.iloc[valid_idx, :]
Y_trainvalid = Y[trainvalid_idx]
Y_valid = Y[valid_idx]


print("calcultaing distance matrix..", ctime())
matrix_euc, matrix_h, matrix_p = cal_matrixs(
    X_trainvalid, X_trainvalid, categorical, non_categorical
)
matrix_euc_x_invitro, matrix_h_x_invitro, matrix_p_x_invitro = cal_matrixs(
    X_trainvalid, db_invitro, categorical_both, non_categorical
)
print("distance matrix calculation finished.", ctime())

# ------------------------hyperparameters range---------
if args.alpha_h == "logspace":
    sequence_ap = np.logspace(-2, 0, 20)
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
# 50
# -------------------training --------------------
best_accs = 0
best_p = dict()

count = 1
print("training process start..", ctime())
grid_search = pd.DataFrame()
for ah in sequence_ah:
    for ap in sequence_ap:
        print(
            "*" * 50,
            count / (len(sequence_ap) ** 2),
            ctime(),
            end="\r",
        )
        count = count + 1

        if args.model == "rf":
            model = RandomForestClassifier(random_state=10)
        elif args.model == "lr":
            model = LogisticRegression(random_state=10)
        # model = LogisticRegression(random_state=10, n_jobs=60)

        result = RASAR_simple(
            df_fishchem_tv,
            col_groups,
            matrix_euc,
            matrix_h,
            matrix_p,
            ah,
            ap,
            X_trainvalid,
            Y_trainvalid,
            db_invitro_matrix=(
                matrix_h_x_invitro,
                matrix_p_x_invitro,
                matrix_euc_x_invitro,
            ),
            invitro=args.w_invitro,
            n_neighbors=args.n_neighbors,
            invitro_form=args.invitro_label,
            db_invitro=db_invitro,
            encoding=args.encoding,
            model=model,
        )

        # testing
        minmax = MinMaxScaler().fit(X_trainvalid[non_categorical])

        X_trainvalid[non_categorical] = minmax.transform(
            X_trainvalid.loc[:, non_categorical]
        )
        X_valid[non_categorical] = minmax.transform(X_valid.loc[:, non_categorical])
        X[non_categorical] = minmax.transform(X.loc[:, non_categorical])
        db_invitro[non_categorical] = minmax.transform(
            db_invitro.loc[:, non_categorical]
        )

        matrix_test = dist_matrix(
            X_valid, X_trainvalid, non_categorical, categorical, ah, ap
        )
        matrix_train = dist_matrix(
            X_trainvalid, X_trainvalid, non_categorical, categorical, ah, ap
        )

        db_invitro_matrix = dist_matrix(
            X, db_invitro, non_categorical, categorical_both, ah, ap
        )

        train_index = X_trainvalid.index
        test_index = X_valid.index
        train_rf, test_rf = cal_s_rasar(
            matrix_train, matrix_test, Y_trainvalid, args.n_neighbors, args.encoding
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
                    train_rf,
                    db_invitro,
                    db_invitro_matrix,
                    train_index,
                    args.invitro_label,
                )
                test_rf = find_nearest_vitro(
                    test_rf,
                    db_invitro,
                    db_invitro_matrix,
                    test_index,
                    args.invitro_label,
                )

        df_test_score = fit_and_predict(
            model, train_rf, Y_trainvalid, test_rf, Y_valid, args.encoding
        )

        df_output = result
        df_mean = pd.DataFrame(df_output.mean(axis=0)).transpose()
        df_std = pd.DataFrame(df_output.sem(axis=0)).transpose()

        temp_grid = pd.DataFrame()
        temp_grid = pd.concat([temp_grid, pd.DataFrame([ah], columns=["ah"])], axis=1)
        temp_grid = pd.concat([temp_grid, pd.DataFrame([ap], columns=["ap"])], axis=1)
        temp_grid = pd.concat(
            [
                temp_grid,
                df_mean.add_prefix("train_avg_"),
                df_std.add_prefix("train_std_"),
                df_test_score.add_prefix("test_"),
            ],
            axis=1,
        )
        grid_search = pd.concat([grid_search, temp_grid])
df2file(grid_search, args.outputFile + "_bestalphas.txt")
best_ah = grid_search.sort_values(by=["train_avg_accuracy"]).ah.values[0]
best_ap = grid_search.sort_values(by=["train_avg_accuracy"]).ap.values[0]

for i in tqdm(range(0, len(params_comb))):

    if args.model == "rf":
        model = RandomForestClassifier(random_state=10)
    elif args.model == "lr":
        model = LogisticRegression(random_state=10)
    # model = LogisticRegression(random_state=10, n_jobs=60)
    for k, v in params_comb[i].items():
        setattr(model, k, v)
    try:
        result = RASAR_simple(
            df_fishchem_tv,
            col_groups,
            matrix_euc,
            matrix_h,
            matrix_p,
            best_ah,
            best_ap,
            X_trainvalid,
            Y_trainvalid,
            db_invitro_matrix=(
                matrix_h_x_invitro,
                matrix_p_x_invitro,
                matrix_euc_x_invitro,
            ),
            invitro=args.w_invitro,
            n_neighbors=args.n_neighbors,
            invitro_form=args.invitro_label,
            db_invitro=db_invitro,
            encoding=args.encoding,
            model=model,
        )

        # -------------------tested on test dataset--------------------
        print("testing start.", ctime())
        minmax = MinMaxScaler().fit(X_trainvalid[non_categorical])
        X_trainvalid[non_categorical] = minmax.transform(
            X_trainvalid.loc[:, non_categorical]
        )
        X_valid[non_categorical] = minmax.transform(X_valid.loc[:, non_categorical])
        X[non_categorical] = minmax.transform(X.loc[:, non_categorical])
        db_invitro[non_categorical] = minmax.transform(
            db_invitro.loc[:, non_categorical]
        )

        matrix_test = dist_matrix(
            X_valid, X_trainvalid, non_categorical, categorical, best_ah, best_ap
        )
        matrix_train = dist_matrix(
            X_trainvalid, X_trainvalid, non_categorical, categorical, best_ah, best_ap
        )

        db_invitro_matrix = dist_matrix(
            X, db_invitro, non_categorical, categorical_both, best_ah, best_ap
        )

        train_index = X_trainvalid.index
        test_index = X_valid.index
        train_rf, test_rf = cal_s_rasar(
            matrix_train, matrix_test, Y_trainvalid, args.n_neighbors, args.encoding
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
                    train_rf,
                    db_invitro,
                    db_invitro_matrix,
                    train_index,
                    args.invitro_label,
                )
                test_rf = find_nearest_vitro(
                    test_rf,
                    db_invitro,
                    db_invitro_matrix,
                    test_index,
                    args.invitro_label,
                )

        # print(train_rf.columns)
        df_test_score = fit_and_predict(
            model, train_rf, Y_trainvalid, test_rf, Y_valid, args.encoding
        )

        df_output = pd.concat(
            [df_mean, df_std, df_test_score],
            keys=["train_mean", "train_std", "test"],
            names=["series_name"],
        )

        df_output = pd.concat(result, axis=0)
        df_mean = pd.DataFrame(df_output.mean(axis=0)).transpose()
        df_std = pd.DataFrame(df_output.sem(axis=0)).transpose()
    except:
        pass
    temp_grid = pd.DataFrame()
    temp_grid = pd.concat([temp_grid, pd.DataFrame([best_ah], columns=["ah"])], axis=1)
    temp_grid = pd.concat([temp_grid, pd.DataFrame([best_ap], columns=["ap"])], axis=1)
    for k, v in params_comb[i].items():
        temp_grid = pd.concat([temp_grid, pd.DataFrame([v], columns=[k])], axis=1)
    temp_grid = pd.concat(
        [
            temp_grid,
            df_mean.add_prefix("train_avg_"),
            df_std.add_prefix("train_std_"),
            df_test_score.add_prefix("test_"),
        ],
        axis=1,
    )

    grid_search = pd.concat([grid_search, temp_grid])
# ----------------save the information into a file-------
df2file(grid_search, args.outputFile + ".txt")


# ----------------------------------------------------------------------------------general: invitro + (invivo) -> invivo (gtv/gtvv)--------------------------------------------
# binary, R=4
# python RASAR_simple_addinginvitro_general_cte.py -i /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -iv /local/wujimeng/code_jimeng/data/invitro/invitro_processed.csv  -wi "True" -il "number" -n 4 -ah 0.23357214690901212 -ap 0.1128837891684889  -o "vitro_e/vivo+vitro/bestR/general/general.txt"
# python RASAR_simple_addinginvitro_general_cte.py -i /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -iv /local/wujimeng/code_jimeng/data/invitro/invitro_processed.csv  -wi "own"  -il "number" -n 4 -ah 0.23357214690901212 -ap 0.1128837891684889  -o "vitro_e/vivo+vitro/bestR/general/general_own_invitro.txt"

# binary, R=1
# python RASAR_simple_addinginvitro_general_cte.py -i /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -iv /local/wujimeng/code_jimeng/data/invitro/invitro_processed.csv  -wi "True" -il "number" -n 1  -o "general/vivo+vitro/general"
# python RASAR_simple_addinginvitro_general_cte.py -i /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -iv /local/wujimeng/code_jimeng/data/invitro/invitro_processed.csv  -wi "own"  -il "number" -n 1  -o "general/vivo+vitro/general_own_invitro"


# multiclass, R=5
# python RASAR_simple_addinginvitro_general_cte.py -i data/invivo/lc50_processed_jim.csv  -i2 data/invitro/invitro_processed.csv -e "multiclass" -wi "True" -il "number" -n 5 -ah 0.04281332398719394 -ap 0.7847599703514611  -o "vitro_e/vivo+vitro/bestR/multiclass/general/general_mul.txt"
# python RASAR_simple_addinginvitro_general_cte.py -i data/invivo/lc50_processed_jim.csv  -i2 data/invitro/invitro_processed.csv -e "multiclass" -wi "own"  -il "number" -n 5 -ah 0.04281332398719394 -ap 0.7847599703514611  -o "vitro_e/vivo+vitro/bestR/multiclass/general/general_own_invitro_mul.txt"
