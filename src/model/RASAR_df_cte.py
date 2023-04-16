from helper_cte_model import *
from sklearn.model_selection import ParameterSampler
import h2o

import argparse

class_threshold = None


def getArguments():
    parser = argparse.ArgumentParser(description="Running KNN_model for datasets.")
    parser.add_argument(
        "-i", "--input", dest="inputFile", help="input file position", required=True
    )
    parser.add_argument("-idf", "--input_df", dest="inputFile_df", required=True)
    parser.add_argument(
        "-iv",
        "--input_vitro",
        dest="inputFile_vitro",
        help="input invitro file position",
        default="no",
    )
    parser.add_argument("-ah", "--alpha_h", dest="hamming_alpha", required=True)
    parser.add_argument("-ap", "--alpha_p", dest="pubchem2d_alpha", required=True)
    parser.add_argument(
        "-dbi",
        "--db_invitro",
        dest="db_invitro",
        help="yes: add in vitro as other source for distance matrix, \
            no: do not use in vitro as input, \
            overlap: use in vitro as input feature",
        default="no",
    )
    parser.add_argument(
        "-e",
        "--encoding",
        dest="encoding",
        help="binary or multiclass (5 class)",
        default="binary",
    )
    parser.add_argument("-effect", "--train_effect", dest="train_effect", required=True)
    parser.add_argument(
        "-endpoint", "--train_endpoint", dest="train_endpoint", required=True
    )
    parser.add_argument(
        "-il",
        "--invitro_label",
        dest="invitro_label",
        help="number, label",
        default="number",
    )
    parser.add_argument(
        "-n", "--n_neighbors", dest="n_neighbors", nargs="?", default=1, type=int
    )
    parser.add_argument(
        "-ni",
        "--niter",
        dest="niter",
        default=50,
        help="model iteration number to find the best hyperparameters",
        type=int,
    )
    parser.add_argument(
        "-wi",
        "--w_invitro",
        dest="w_invitro",
        help="own:in vitro alone as input  , \
            false:in vitro not as input ,\
            true:use in vitro and in vivo as input",
        default="False",
    )
    parser.add_argument("-o", "--output", dest="outputFile", default="binary.txt")
    return parser.parse_args()


args = getArguments()
# ---------------info for running the model----------------
if args.encoding == "binary":
    encoding = "binary"
    encoding_value = 1
elif args.encoding == "multiclass":
    encoding = "multiclass"
    encoding_value = [0.1, 1, 10, 100]

# -----------loading data & splitting into train and test dataset--------
print("loading dataset...", ctime())
if args.db_invitro == "no" or args.db_invitro == "overlap":
    db_mortality, db_datafusion = load_datafusion_datasets(
        args.inputFile,
        args.inputFile_df,
        categorical_columns=categorical,
        encoding=encoding,
        encoding_value=encoding_value,
    )
    db_invitro = args.db_invitro
else:
    db_mortality, db_datafusion, db_invitro = load_datafusion_datasets_invitro(
        args.inputFile,
        args.inputFile_df,
        args.inputFile_vitro,
        categorical_columns=categorical,
        encoding=encoding,
        encoding_value=encoding_value,
    )
    # db_invitro = vitroconc_to_label(
    #     db_invitro, db_invitro.invitro_conc.median()
    # )  # labelling the in vitro concentration

print("Data loaded.", ctime())
# -----------------startified split the dataset for training model--------

test_size = 0.2
col_groups = "test_cas"

df_fishchem = db_mortality[["fish", "test_cas"]]
traintest_idx, valid_idx = get_grouped_train_test_split(
    df_fishchem, test_size, col_groups
)
df_fishchem_tv = df_fishchem.iloc[traintest_idx, :]


X = db_mortality.drop(columns="conc1_mean").copy()
Y = db_mortality.conc1_mean.values

X_traintest = X.iloc[traintest_idx, :]
X_valid = X.iloc[valid_idx, :]
Y_traintest = Y[traintest_idx]
Y_valid = Y[valid_idx]


# -----------------creating the distance matrix--------

matrices = cal_matrixs(X_traintest, X_traintest, categorical, non_categorical)
matrices_df = cal_matrixs(
    X_traintest,
    db_datafusion.drop(columns="conc1_mean").copy(),
    categorical,
    non_categorical,
)
if args.db_invitro != "no" and args.db_invitro != "overlap":
    matrices_invitro = cal_matrixs(
        X_traintest, db_invitro, categorical_both, non_categorical
    )
else:
    matrices_invitro = None

print("distance matrix successfully calculated!", ctime())


# ------------------------hyperparameters range---------

if encoding == "binary":
    model = RandomForestClassifier(random_state=10)
    # hyper_params_tune = {
    #     "max_depth": [i for i in range(10, 20, 2)],
    #     "n_estimators": [int(x) for x in np.linspace(start=100, stop=1000, num=2)],
    #     "min_samples_split": [2, 5, 10],
    #     "min_samples_leaf": [1, 2, 4],
    # }
    hyper_params_tune = {
        "n_estimators": [int(x) for x in np.linspace(start=200, stop=500, num=6)],
        "max_depth": [i for i in range(10, 36, 5)],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 4, 8, 16],
    }

elif encoding == "multiclass":
    h2o.init()
    h2o.no_progress()
    model = H2ORandomForestEstimator(seed=10)
    hyper_params_tune = {
        "ntrees": [i for i in range(10, 1000, 10)],
        "max_depth": [i for i in range(10, 1000, 10)],
        "min_rows": [1, 10, 100, 1000],
        "sample_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    }
rand = 2
params_comb = list(ParameterSampler(hyper_params_tune, n_iter=args.niter))


if args.hamming_alpha == "logspace":
    sequence_ah = np.logspace(-2, 0, 30)
else:
    sequence_ah = [float(args.hamming_alpha)]

if args.pubchem2d_alpha == "logspace":
    sequence_ap = np.logspace(-2, 0, 30)
else:
    sequence_ap = [float(args.pubchem2d_alpha)]

# -------------------training using the default model setting and found the best alphas combination--------------------

best_accs = 0

count = 1
for ah in sequence_ah:
    for ap in sequence_ap:
        for i in range(0, len(params_comb)):
            for k, v in params_comb[i].items():
                setattr(model, k, v)
            print(
                ah,
                ap,
                "*" * 50,
                count / (len(sequence_ap) * len(sequence_ap) * len(params_comb)),
                ctime(),
                end="\r",
            )
            results = cv_datafusion_rasar(
                matrices,
                matrices_df,
                matrices_invitro=matrices_invitro,
                ah=ah,
                ap=ap,
                X=X_traintest,
                Y=Y_traintest,
                db_datafusion=db_datafusion,
                db_invitro=db_invitro,
                train_endpoint=args.train_endpoint,
                train_effect=args.train_effect,
                df_fishchem_tv=df_fishchem_tv,
                col_groups=col_groups,
                model=model,
                n_neighbors=args.n_neighbors,
                invitro=args.w_invitro,
                invitro_form=args.invitro_label,
                encoding=encoding,
            )
            # print(results["avg_accs"])
            if np.mean(results.accuracy) > best_accs:
                best_param = params_comb[i]
                best_accs = np.mean(results.accuracy)
                best_result = results
                best_ah = ah
                best_ap = ap
                print("success.", best_accs)
            count = count + 1


df_mean = pd.DataFrame(best_result.mean(axis=0)).transpose()
df_std = pd.DataFrame(best_result.sem(axis=0)).transpose()


# -------------------tested on test dataset--------------------

print("start testing...", ctime())
for k, v in best_param.items():
    setattr(model, k, v)

X = db_mortality.drop(columns="conc1_mean")

minmax = MinMaxScaler().fit(X_traintest[non_categorical])
X_traintest[non_categorical] = minmax.transform(X_traintest.loc[:, non_categorical])
X_valid[non_categorical] = minmax.transform(X_valid.loc[:, non_categorical])
X[non_categorical] = minmax.transform(X.loc[:, non_categorical])
db_datafusion[non_categorical] = minmax.transform(db_datafusion.loc[:, non_categorical])

matrix_valid = dist_matrix(
    X_valid, X_traintest, non_categorical, categorical, best_ah, best_ap
)
matrix_traintest = dist_matrix(
    X_traintest, X_traintest, non_categorical, categorical, best_ah, best_ap
)
db_datafusion_matrix = dist_matrix(
    X, db_datafusion, non_categorical, categorical, best_ah, best_ap
)

if args.db_invitro != "no" and args.db_invitro != "overlap":

    db_invitro[non_categorical] = minmax.transform(db_invitro.loc[:, non_categorical])

    db_invitro_matrix_train = dist_matrix(
        X_traintest, db_invitro, non_categorical, categorical_both, best_ah, best_ap
    )
    db_invitro_matrix_test = dist_matrix(
        X_valid, db_invitro, non_categorical, categorical_both, best_ah, best_ap
    )


s_rasar_traintest, s_rasar_valid = cal_s_rasar(
    matrix_traintest, matrix_valid, Y_traintest, args.n_neighbors, args.encoding
)

df_rasar_traintest, df_rasar_valid = cal_df_rasar(
    traintest_idx,
    valid_idx,
    X_traintest,
    X_valid,
    db_datafusion,
    db_datafusion_matrix,
    args.train_endpoint,
    args.train_effect,
    encoding,
)

traintest_rf = pd.concat([s_rasar_traintest, df_rasar_traintest], axis=1)
valid_rf = pd.concat([s_rasar_valid, df_rasar_valid], axis=1)

if args.w_invitro == "own":
    traintest_rf = pd.DataFrame()
    valid_rf = pd.DataFrame()

if args.w_invitro != "False":
    if str(args.db_invitro) == "overlap":
        traintest_rf = get_vitroinfo(traintest_rf, X, traintest_idx, args.invitro_label)
        valid_rf = get_vitroinfo(valid_rf, X, valid_idx, args.invitro_label)
    else:
        db_invitro_matrix = dist_matrix(
            X, db_invitro, non_categorical, categorical_both, best_ah, best_ap
        )
        traintest_rf = find_nearest_vitro(
            traintest_rf,
            db_invitro,
            db_invitro_matrix,
            traintest_idx,
            args.invitro_label,
        )
        valid_rf = find_nearest_vitro(
            valid_rf, db_invitro, db_invitro_matrix, valid_idx, args.invitro_label,
        )

if encoding == "binary":
    df_test_score = fit_and_predict(
        model, traintest_rf, Y_traintest, valid_rf, Y_valid, encoding,
    )

elif encoding == "multiclass":

    traintest_rf.loc[:, "target"] = Y_traintest
    valid_rf.loc[:, "target"] = Y_valid

    train_rf_h2o = h2o.H2OFrame(traintest_rf)
    test_rf_h2o = h2o.H2OFrame(valid_rf)

    for col in traintest_rf.columns:
        if "label" in col:
            train_rf_h2o[col] = train_rf_h2o[col].asfactor()
            test_rf_h2o[col] = test_rf_h2o[col].asfactor()

    train_rf_h2o["target"] = train_rf_h2o["target"].asfactor()
    test_rf_h2o["target"] = test_rf_h2o["target"].asfactor()

    model.train(y="target", training_frame=train_rf_h2o)
    y_pred = model.predict(test_rf_h2o).as_data_frame()["predict"]

    df_test_score = pd.DataFrame()
    df_test_score.loc[0, "accuracy"] = accuracy_score(Y_valid, y_pred)
    df_test_score.loc[0, "recall"] = recall_score(Y_valid, y_pred, average="macro")
    df_test_score.loc[0, "specificity"] = np.nan
    df_test_score.loc[0, "f1"] = f1_score(Y_valid, y_pred, average="macro")
    df_test_score.loc[0, "precision"] = precision_score(
        Y_valid, y_pred, average="macro"
    )
    h2o.shutdown()


for k, v in best_param.items():
    df_test_score.loc[0, k] = str(v)

df_test_score.loc[0, ["ah", "ap"]] = best_result.iloc[0][["ah", "ap"]]
df_test_score.loc[0, "encoding"] = args.encoding

df_output = pd.concat(
    [df_mean, df_std, df_test_score],
    keys=["train_mean", "train_std", "test"],
    names=["series_name"],
)


# ----------------save the information into a file-------
df2file(df_output, args.outputFile)

