from helper_cte_model import *
from sklearn.model_selection import ParameterSampler
import h2o
import argparse


def getArguments():
    parser = argparse.ArgumentParser(description="Running datafusion RASAR model for stratified splitted datasets, \
                                     with not only information from mortality, but also from other effects. and the \
                                     model was first trained to select the best alphas combination, then we selected the \
                                     best hyperparameter combination.")
    parser.add_argument("-i", "--input", dest="inputFile", required=True)
    parser.add_argument("-idf", "--input_df", dest="inputFile_df", required=True)
    parser.add_argument("-iv", "--input_vitro", dest="inputFile_vitro", default="no")
    parser.add_argument("-ah", "--alpha_h", dest="alpha_h", required=True, nargs="?")
    parser.add_argument("-ap", "--alpha_p", dest="alpha_p", nargs="?")
    parser.add_argument("-e", "--encoding", dest="encoding", default="binary")
    parser.add_argument("-effect", "--train_effect", dest="train_effect", default="MOR")
    parser.add_argument(
        "-endpoint", "--train_endpoint", dest="train_endpoint", required=True
    )
    parser.add_argument(
        "-il", "--invitro_label", dest="invitro_label", default="number"
    )
    parser.add_argument(
        "-n", "--n_neighbors", dest="n_neighbors", nargs="?", default=1, type=int
    )
    parser.add_argument("-ni", "--niter", dest="niter", default=20, type=int)
    parser.add_argument("-wi", "--w_invitro", dest="w_invitro", default="False")

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

rand = random.randrange(1, 100)

test_size = 0.2

col_groups = "test_cas"

invitro_matrices = None

# ----------loading dataset--------------
print("loading dataset...", ctime())
db_mortality, db_datafusion = load_datafusion_datasets(
    args.inputFile,
    args.inputFile_df,
    categorical_columns=categorical,
    encoding=encoding,
    encoding_value=encoding_value,
)
print("finish loaded.", ctime())
# ------------------ stratified split the dataset into train and test---------
df_fishchem = db_mortality[["fish", "test_cas"]]

traintest_idx, valid_idx = get_grouped_train_test_split(
    df_fishchem, test_size, col_groups
)

df_fishchem_tv = df_fishchem.iloc[traintest_idx, :].reset_index(drop=True)

X = db_mortality.drop(columns="conc1_mean").copy()
X_traintest = db_mortality.drop(columns="conc1_mean").iloc[traintest_idx, :]
X_valid = db_mortality.drop(columns="conc1_mean").iloc[valid_idx, :]
Y_traintest = db_mortality.iloc[traintest_idx, :].conc1_mean.values
Y_valid = db_mortality.iloc[valid_idx, :].conc1_mean.values
print("Data loaded.", ctime())


# ----------------------distance map creating-------------
print("calcultaing distance matrix..", ctime())
matrices = cal_matrixs(X_traintest, X_traintest, categorical, non_categorical)
matrices_df = cal_matrixs(
    X_traintest,
    db_datafusion.drop(columns="conc1_mean").copy(),
    categorical,
    non_categorical,
)
print("distance matrix successfully calculated!", ctime())

del db_mortality

# --------------model hyperparameter range-------------------
if encoding == "binary":
    model = RandomForestClassifier()
    # hyper_params_tune = {
    #     "max_depth": [i for i in range(10, 30, 6)],
    #     "n_estimators": [int(x) for x in np.linspace(start=200, stop=1000, num=11)],
    #     "min_samples_split": [2, 5, 10],
    #     "min_samples_leaf": [1, 2, 4, 8, 16, 32],
    # }
    hyper_params_tune = {
        "n_estimators": [int(x) for x in np.linspace(start=200, stop=500, num=6)],
        "max_depth": [i for i in range(10, 100, 10)],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 4, 8, 16],
        "class_weight": [{0: i, 1: 1} for i in range(0, 20)],
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
params_comb = list(
    ParameterSampler(hyper_params_tune, n_iter=args.niter, random_state=rand)
)


if args.alpha_h == "logspace":
    sequence_ap = np.logspace(-2, 0, 2)
    sequence_ah = sequence_ap
else:
    sequence_ap = [float(args.alpha_p)]
    sequence_ah = [float(args.alpha_h)]


# -------------------training using the default model setting and found the best alphas combination--------------------

best_accs = 0
best_param = dict()
count = 1
num_runs = len(sequence_ap) ** 2 + len(params_comb)
starttime = datetime.datetime.now()

for ah in sequence_ah:
    for ap in sequence_ap:
        remain_time = (datetime.datetime.now() - starttime) / count * (num_runs - count)
        print(
            ah,
            ap,
            "*" * 50,
            count / num_runs,
            remain_time,
            "remained.",
            ctime(),
            end="\r",
        )
        results = cv_datafusion_rasar(
            matrices,
            matrices_df,
            db_invitro_matrix=invitro_matrices,
            ah=ah,
            ap=ap,
            X=X_traintest,
            Y=Y_traintest,
            db_datafusion=db_datafusion,
            db_invitro=args.inputFile_vitro,
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

        if np.mean(results.accuracy) > best_accs:
            best_accs = np.mean(results.accuracy)
            best_result = results
            best_ah = ah
            best_ap = ap
            print("success.", best_accs, end="\r")

        count = count + 1

# --------------explore more on the model's hyperparameter range with found alphas--------
for i in range(0, len(params_comb)):
    remain_time = (datetime.datetime.now() - starttime) / count * (num_runs - count)
    print(
        best_ah,
        best_ap,
        "*" * 50,
        count / num_runs,
        remain_time,
        "remained.",
        ctime(),
        end="\r",
    )
    try:
        for k, v in params_comb[i].items():
            setattr(model, k, v)

        results = cv_datafusion_rasar(
            matrices,
            matrices_df,
            db_invitro_matrix="no",
            ah=best_ah,
            ap=best_ap,
            X=X_traintest,
            Y=Y_traintest,
            db_datafusion=db_datafusion,
            db_invitro=args.inputFile_vitro,
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

        if np.mean(results.accuracy) > best_accs:
            best_param = params_comb[i]
            best_accs = np.mean(results.accuracy)
            best_result = results
            print("success.", best_accs, end="\r")

    except:
        pass

    count = count + 1

df_mean = pd.DataFrame(best_result.mean(axis=0)).transpose()
df_std = pd.DataFrame(best_result.sem(axis=0)).transpose()

# -------------------tested on test dataset--------------------
print("start testing...", ctime())
for k, v in best_param.items():
    setattr(model, k, v)

traintest_index = X_traintest.index
valid_index = X_valid.index

minmax = MinMaxScaler().fit(X_traintest[non_categorical])
X_traintest[non_categorical] = minmax.transform(X_traintest.loc[:, non_categorical])
X_valid[non_categorical] = minmax.transform(X_valid.loc[:, non_categorical])
X[non_categorical] = minmax.transform(X.loc[:, non_categorical])
db_datafusion[non_categorical] = minmax.transform(db_datafusion.loc[:, non_categorical])


matrices_valid = dist_matrix(
    X_valid, X_traintest, non_categorical, categorical, best_ah, best_ap,
)
matrices_traintest = dist_matrix(
    X_traintest, X_traintest, non_categorical, categorical, best_ah, best_ap,
)
db_datafusion_matrix = dist_matrix(
    X,
    db_datafusion.drop(columns="conc1_mean").copy(),
    non_categorical,
    categorical,
    best_ah,
    best_ap,
)


s_rasar_traintest, s_rasar_valid = cal_s_rasar(
    matrices_traintest, matrices_valid, Y_traintest, args.n_neighbors, encoding,
)

df_rasar_traintest, df_rasar_valid = cal_df_rasar(
    traintest_index,
    valid_index,
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
    if str(args.inputFile_vitro) == "overlap":
        traintest_rf = get_vitroinfo(
            traintest_rf, X_traintest, traintest_index, args.invitro_label
        )
        valid_rf = get_vitroinfo(valid_rf, X_valid, valid_index, args.invitro_label)

if encoding == "binary":
    df_test_score = fit_and_predict(
        model, traintest_rf, Y_traintest, valid_rf, Y_valid, encoding,
    )

elif encoding == "multiclass":

    traintest_rf.loc[:, "target"] = Y_traintest.values
    valid_rf.loc[:, "target"] = Y_valid.values

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
# python RASAR_df_cte.py       -i  "C:/Users/wjmen/Desktop/GitHub/data/invivo/lc50_test.csv"  -idf  "C:/Users/wjmen/Desktop/GitHub/data/invivo/df_test.csv"      -endpoint ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o test/df_rasar.txt -ni 5


# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv      -endpoint ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o effects/R1_rasar_alphas/binary/df_rasar.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ACC.csv  -endpoint ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o effects/R1_rasar_alphas/binary/df_rasar_ACC.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BCM.csv  -endpoint ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o effects/R1_rasar_alphas/binary/df_rasar_BCM.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BEH.csv  -endpoint ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o effects/R1_rasar_alphas/binary/df_rasar_BEH.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_CEL.csv  -endpoint ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o effects/R1_rasar_alphas/binary/df_rasar_CEL.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ENZ.csv  -endpoint ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o effects/R1_rasar_alphas/binary/df_rasar_ENZ.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_GEN.csv  -endpoint ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o effects/R1_rasar_alphas/binary/df_rasar_GEN.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ITX.csv  -endpoint ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o effects/R1_rasar_alphas/binary/df_rasar_ITX.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_PHY.csv  -endpoint ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o effects/R1_rasar_alphas/binary/df_rasar_PHY.txt


# ------------------------------------------------------invivo to invivo(gvvdf)----------------------------------------------------------
# binary, datafusion, R=4
# python RASAR_df_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv     -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv -endpoint ['LC50','EC50'] -effect 'MOR'  -ah 0.23357214690901212 -ap 0.11288378916846889 -n 4 -o vitro_e/vivo+vitro/general/vivo_df_rasar_bi_4.csv
# python RASAR_df_cte.py -i1 data/invivo/lc50_processed_jim.csv     -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv -endpoint ['LC50','EC50'] -effect 'MOR'  -ah 0.041753189365604   -ap 4.893900918477494        -o rasar/df_rasar.txt


# ------------------------------------------------------invivo to invivo(ovvdf)----------------------------------------------------------
# python RASAR_df_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro.csv -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv -endpoint ['LC50','EC50'] -effect 'MOR'  -ah logspace -ap logspace -dbi "no"      -wi "False" -il "both" -o vitro_e/vivo/vivo_df_rasar_bi_repeated_4.csv

# ------------------------------------------------------invitro + invivo to invivo(otvvdf)----------------------------------------------------------
# python RASAR_df_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro.csv -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv -endpoint ['LC50','EC50'] -effect 'MOR'  -ah logspace -ap logspace -dbi "overlap" -wi "True"  -il "number" -o vitro_e/vivo+vitro/bestR/repeat_number_df.txt
# python RASAR_df_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro.csv -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv -endpoint ['LC50','EC50'] -effect 'MOR'  -ah logspace -ap logspace -dbi "overlap" -wi "True"  -il "label"  -o vitro_e/vivo+vitro/bestR/repeat_label_df.txt
# python RASAR_df_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro.csv -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv -endpoint ['LC50','EC50'] -effect 'MOR'  -ah logspace -ap logspace -dbi "overlap" -wi "True"  -il "both"   -o vitro_e/vivo+vitro/bestR/repeat_both_df.txt
# python RASAR_df_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro.csv -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv -endpoint ['LC50','EC50'] -effect 'MOR'  -ah logspace -ap logspace -dbi "overlap" -wi "own"   -il "number" -o vitro_e/vivo+vitro/bestR/repeat_own_number_df.txt
# python RASAR_df_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro.csv -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv -endpoint ['LC50','EC50'] -effect 'MOR'  -ah logspace -ap logspace -dbi "overlap" -wi "own"   -il "label"  -o vitro_e/vivo+vitro/bestR/repeat_own_label_df.txt
# python RASAR_df_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro.csv -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv -endpoint ['LC50','EC50'] -effect 'MOR'  -ah logspace -ap logspace -dbi "overlap" -wi "own"   -il "both"   -o vitro_e/vivo+vitro/bestR/repeat_own_both_df.txt

# -ah 0.01 -ap 0.15264179671752334
