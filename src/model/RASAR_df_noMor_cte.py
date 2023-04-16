<<<<<<< HEAD:src/model/RASAR_df_noMor_cte.py
from helper_cte_model import *
from sklearn.model_selection import ParameterSampler
import h2o
import argparse
import random


def getArguments():
    parser = argparse.ArgumentParser(description="Running datafusion RASAR model for stratified splitting datasets, \
                                    without the input information from the mortality dataset but only information from \
                                     other effects or in vitro.")
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
        "-fixed", "--fixed_threshold", dest="fixed_threshold", default="no"
    )
    parser.add_argument("-ni", "--niter", dest="niter", default=20, type=int)
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

rand = random.randrange(1, 100)
test_size = 0.2
col_groups = "test_cas"


# -----------loading data & splitting into train and test dataset--------
print("loading dataset...", ctime())
db_mortality, db_datafusion = load_datafusion_datasets(
    args.inputFile,
    args.inputFile_df,
    categorical_columns=categorical,
    encoding=encoding,
    encoding_value=encoding_value,
)
print("finish loaded.", ctime())


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

Y_traintest = db_mortality.iloc[traintest_idx, :].conc1_mean.values
Y_valid = db_mortality.iloc[valid_idx, :].conc1_mean.values

print("calcultaing distance matrix..", ctime())
matrices = cal_matrixs(X_traintest, X_traintest, categorical, non_categorical)
matrices_df = cal_matrixs(
    X_traintest,
    db_datafusion.drop(columns="conc1_mean").copy(),
    categorical,
    non_categorical,
)
print("distance matrix successfully calculated!", ctime())

if encoding == "binary":
    model = RandomForestClassifier()
    hyper_params_tune = {
        "max_depth": [i for i in range(10, 100, 10)],
        "n_estimators": [int(x) for x in np.linspace(start=200, stop=1000, num=11)],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4, 8, 16, 32],
    }
elif encoding == "multiclass":
    h2o.init()
    h2o.no_progress()
    model = H2ORandomForestEstimator(seed=rand)
    hyper_params_tune = {
        "ntrees": [i for i in range(10, 1000, 10)],
        "max_depth": [i for i in range(10, 1000, 10)],
        "min_rows": [1, 10, 100, 1000],
        "sample_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "class_weight": [{0: i, 1: 1} for i in range(0, 20)],
    }


params_comb = list(
    ParameterSampler(hyper_params_tune, n_iter=args.niter, random_state=rand)
)

if args.alpha_h == "logspace":
    sequence_ap = np.logspace(-2, 0, 20)
    sequence_ah = sequence_ap
else:
    sequence_ap = [float(args.alpha_p)]
    sequence_ah = [float(args.alpha_h)]


best_accs = 0
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
            matrices_invitro="no",
            ah=ah,
            ap=ap,
            X=X_traintest,
            Y=Y_traintest,
            db_datafusion=db_datafusion,
            db_invitro="no",
            train_endpoint=args.train_endpoint,
            train_effect=args.train_effect,
            df_fishchem_tv=df_fishchem_tv,
            col_groups=col_groups,
            model=model,
            invitro="False",
            invitro_form="no",
            encoding=encoding,
            wmor=False,
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

    for k, v in params_comb[i].items():
        setattr(model, k, v)

    results = cv_datafusion_rasar(
        matrices,
        matrices_df,
        matrices_invitro="no",
        ah=best_ah,
        ap=best_ap,
        X=X_traintest,
        Y=Y_traintest,
        db_datafusion=db_datafusion,
        db_invitro="no",
        train_endpoint=args.train_endpoint,
        train_effect=args.train_effect,
        df_fishchem_tv=df_fishchem_tv,
        col_groups=col_groups,
        model=model,
        invitro="False",
        invitro_form="no",
        encoding=encoding,
        wmor=False,
    )
    if np.mean(results.accuracy) > best_accs:
        best_param = params_comb[i]
        best_accs = np.mean(results.accuracy)
        best_result = results
        print("success.", best_accs, end="\r")


df_mean = pd.DataFrame(best_result.mean(axis=0)).transpose()
df_std = pd.DataFrame(best_result.sem(axis=0)).transpose()


# -------------------tested on test dataset--------------------
print("start testing...", ctime())
for k, v in best_param.items():
    setattr(model, k, v)

traintest_index = X_traintest.index
valid_index = X_valid.index


X = db_mortality.drop(columns="conc1_mean").copy()

minmax = MinMaxScaler().fit(X_traintest[non_categorical])
X_traintest[non_categorical] = minmax.transform(X_traintest.loc[:, non_categorical])
X_valid[non_categorical] = minmax.transform(X_valid.loc[:, non_categorical])
X[non_categorical] = minmax.transform(X.loc[:, non_categorical])
db_datafusion[non_categorical] = minmax.transform(db_datafusion.loc[:, non_categorical])

db_datafusion_matrix = dist_matrix(
    X,
    db_datafusion.drop(columns="conc1_mean").copy(),
    non_categorical,
    categorical,
    best_ah,
    best_ap,
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

traintest_rf = df_rasar_traintest
test_rf = df_rasar_valid

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
        model, traintest_rf, Y_traintest, test_rf, Y_valid, encoding,
    )
elif encoding == "multiclass":

    traintest_rf.loc[:, "target"] = Y_traintest.values
    test_rf.loc[:, "target"] = Y_valid.values

    train_rf_h2o = h2o.H2OFrame(traintest_rf)
    test_rf_h2o = h2o.H2OFrame(test_rf)

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


# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv      -label ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o effect/R1_rasar_alphas/binary/df_rasar_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ACC.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o effect/R1_rasar_alphas/binary/df_rasar_ACC_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BCM.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o effect/R1_rasar_alphas/binary/df_rasar_BCM_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BEH.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o effect/R1_rasar_alphas/binary/df_rasar_BEH_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_CEL.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o effect/R1_rasar_alphas/binary/df_rasar_CEL_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ENZ.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o effect/R1_rasar_alphas/binary/df_rasar_ENZ_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_GEN.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o effect/R1_rasar_alphas/binary/df_rasar_GEN_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ITX.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o effect/R1_rasar_alphas/binary/df_rasar_ITX_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_PHY.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o effect/R1_rasar_alphas/binary/df_rasar_PHY_noMOR.txt


# NEW
# python RASAR_simple.py   -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv   -e "multiclass" -ah 0.01 -ap 4.832930238571752  -o rasar/R1_knn_alphas/multiclass/s_rasar_mul.txt
# python RASAR_df_noMor.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv      -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_noMOR_mul.txt
# python RASAR_df_noMor.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ACC.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_ACC_noMOR_mul.txt
# python RASAR_df_noMor.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BCM.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_BCM_noMOR_mul.txt
# python RASAR_df_noMor.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BEH.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_BEH_noMOR_mul.txt
# python RASAR_df_noMor.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_CEL.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_CEL_noMOR_mul.txt
# python RASAR_df_noMor.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ENZ.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_ENZ_noMOR_mul.txt
# python RASAR_df_noMor.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_GEN.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_GEN_noMOR_mul.txt
# python RASAR_df_noMor.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ITX.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_ITX_noMOR_mul.txt
# python RASAR_df_noMor.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_PHY.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_PHY_noMOR_mul.txt
# python RASAR_df.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv      -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_mul.txt
# python RASAR_df.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ACC.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_ACC_mul.txt
# python RASAR_df.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BCM.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_BCM_mul.txt
# python RASAR_df.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BEH.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_BEH_mul.txt
# python RASAR_df.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_CEL.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_CEL_mul.txt
# python RASAR_df.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ENZ.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_ENZ_mul.txt
# python RASAR_df.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_GEN.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_GEN_mul.txt
# python RASAR_df.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ITX.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_ITX_mul.txt
# python RASAR_df.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_PHY.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_PHY_mul.txt


# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv      -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_noMOR.txt
# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ACC.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_ACC_noMOR.txt
# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BCM.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_BCM_noMOR.txt
# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BEH.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_BEH_noMOR.txt
# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_CEL.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_CEL_noMOR.txt
# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ENZ.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_ENZ_noMOR.txt
# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_GEN.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_GEN_noMOR.txt
# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ITX.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_ITX_noMOR.txt
# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_PHY.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_PHY_noMOR.txt
# python RASAR_df.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv      -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar.txt
# python RASAR_df.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ACC.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_ACC.txt
# python RASAR_df.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BCM.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_BCM.txt
# python RASAR_df.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BEH.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_BEH.txt
# python RASAR_df.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_CEL.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_CEL.txt
# python RASAR_df.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ENZ.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_ENZ.txt
# python RASAR_df.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_GEN.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_GEN.txt
# python RASAR_df.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ITX.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_ITX.txt
# python RASAR_df.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_PHY.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_PHY.txt
#
# knn binary
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv      -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ACC.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_ACC.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BCM.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_BCM.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BEH.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_BEH.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_CEL.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_CEL.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ENZ.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_ENZ.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_GEN.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_GEN.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ITX.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_ITX.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_PHY.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_PHY.txt

# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv      -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ACC.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_ACC_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BCM.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_BCM_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BEH.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_BEH_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_CEL.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_CEL_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ENZ.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_ENZ_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_GEN.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_GEN_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ITX.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_ITX_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_PHY.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_PHY_noMOR.txt


# rasar binary
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv      -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ACC.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_ACC.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BCM.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_BCM.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BEH.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_BEH.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_CEL.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_CEL.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ENZ.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_ENZ.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_GEN.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_GEN.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ITX.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_ITX.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_PHY.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_PHY.txt

# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv      -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ACC.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_ACC_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BCM.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_BCM_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BEH.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_BEH_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_CEL.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_CEL_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ENZ.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_ENZ_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_GEN.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_GEN_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ITX.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_ITX_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_PHY.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_PHY_noMOR.txt


# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv      -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_noMOR_mul.txt
# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv      -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_noMOR.txt


def cv_datafusion_rasar_womor(
    matrices_df,
    db_invitro_matrix,
    ah,
    ap,
    X,
    Y,
    db_datafusion,
    db_invitro,
    train_endpoint,
    train_effect,
    df_fishchem_tv,
    col_groups,
    model=RandomForestClassifier(),
    invitro=False,
    invitro_form="both",
    encoding="binary",
):
    group_kfold = GroupKFold(n_splits=5)

    list_df_output = []
    for train_index, test_index in group_kfold.split(
        df_fishchem_tv, groups=df_fishchem_tv[col_groups]
    ):
        x_train = X.iloc[train_index]
        x_test = X.iloc[test_index]

        minmax = MinMaxScaler().fit(x_train[non_categorical])
        x_train[non_categorical] = minmax.transform(x_train.loc[:, non_categorical])
        x_test[non_categorical] = minmax.transform(x_test.loc[:, non_categorical])

        X_normed = X.copy()
        X_normed[non_categorical] = minmax.transform(X_normed.loc[:, non_categorical])

        db_datafusion_normed = db_datafusion.copy()
        db_datafusion_normed[non_categorical] = minmax.transform(
            db_datafusion_normed.loc[:, non_categorical]
        )

        db_datafusion_matrix = pd.DataFrame(
            ah * matrices_df["hamming"]
            + ap * matrices_df["pubchem"]
            + euclidean_matrix(X_normed, db_datafusion_normed, non_categorical,)
        )

        y_train = Y.iloc[train_index]
        y_test = Y.iloc[test_index]

        df_rasar_train, df_rasar_test = cal_df_rasar(
            train_index,
            test_index,
            X_normed.iloc[train_index],
            X_normed.iloc[test_index],
            db_datafusion_normed,
            db_datafusion_matrix,
            train_endpoint,
            train_effect,
            encoding,
        )

        train_rf = df_rasar_train
        test_rf = df_rasar_test

        if invitro == "own":
            train_rf = pd.DataFrame()
            test_rf = pd.DataFrame()

        if invitro != "False":
            if str(db_invitro) == "overlap":
                train_rf = get_vitroinfo(train_rf, X_normed, train_index, invitro_form)
                test_rf = get_vitroinfo(test_rf, X_normed, test_index, invitro_form)

                if (invitro != "False") & (invitro_form == "number"):
                    train_rf["invitro_conc"] = X_normed.ec50[train_index].reset_index(
                        drop=True
                    )
                    test_rf["invitro_conc"] = X_normed.ec50[test_index].reset_index(
                        drop=True
                    )

                elif (invitro != "False") & (invitro_form == "label"):
                    train_rf["invitro_label"] = X_normed.invitro_label[
                        train_index
                    ].reset_index(drop=True)
                    test_rf["invitro_label"] = X_normed.invitro_label[
                        test_index
                    ].reset_index(drop=True)

                elif (invitro != "False") & (invitro_form == "both"):
                    train_rf["ec50"] = X_normed.ec50[train_index].reset_index(drop=True)
                    test_rf["ec50"] = X_normed.ec50[test_index].reset_index(drop=True)
                    train_rf["invitro_label"] = X_normed.invitro_label[
                        train_index
                    ].reset_index(drop=True)
                    test_rf["invitro_label"] = X_normed.invitro_label[
                        test_index
                    ].reset_index(drop=True)
            else:
                if (invitro != "False") & (invitro_form == "number"):
                    dist = np.array(db_invitro_matrix.iloc[train_index, :].min(axis=1))
                    ls = np.array(db_invitro_matrix.iloc[train_index, :].idxmin(axis=1))
                    conc = db_invitro["ec50"][ls]
                    dist = db_invitro_matrix.lookup(
                        pd.Series(ls).index, pd.Series(ls).values
                    )
                    train_rf["invitro_conc"] = np.array(conc)
                    train_rf["invitro_dist"] = dist

                    dist = np.array(db_invitro_matrix.iloc[test_index, :].min(axis=1))
                    ls = np.array(db_invitro_matrix.iloc[test_index, :].idxmin(axis=1))
                    conc = db_invitro["ec50"][ls]
                    dist = db_invitro_matrix.lookup(
                        pd.Series(ls).index, pd.Series(ls).values
                    )
                    test_rf["invitro_conc"] = np.array(conc)
                    test_rf["invitro_dist"] = dist

                elif (invitro != "False") & (invitro_form == "label"):
                    dist = np.array(db_invitro_matrix.iloc[train_index, :].min(axis=1))
                    ls = np.array(db_invitro_matrix.iloc[train_index, :].idxmin(axis=1))
                    label = db_invitro["invitro_label"][ls]
                    dist = db_invitro_matrix.lookup(
                        pd.Series(ls).index, pd.Series(ls).values
                    )
                    train_rf["invitro_label"] = np.array(label)
                    train_rf["invitro_dist"] = dist

                    dist = np.array(db_invitro_matrix.iloc[test_index, :].min(axis=1))
                    ls = np.array(db_invitro_matrix.iloc[test_index, :].idxmin(axis=1))
                    label = db_invitro["invitro_label"][ls]
                    dist = db_invitro_matrix.lookup(
                        pd.Series(ls).index, pd.Series(ls).values
                    )
                    test_rf["invitro_label"] = np.array(label)
                    test_rf["invitro_dist"] = dist

                elif (invitro != "False") & (invitro_form == "both"):

                    dist = np.array(db_invitro_matrix.iloc[train_index, :].min(axis=1))
                    ls = np.array(db_invitro_matrix.iloc[train_index, :].idxmin(axis=1))
                    conc = db_invitro["ec50"][ls]
                    label = db_invitro["invitro_label"][ls]
                    train_rf["invitro_conc"] = np.array(conc)
                    train_rf["invitro_label"] = np.array(label)
                    train_rf["invitro_dist"] = dist

                    dist = np.array(db_invitro_matrix.iloc[test_index, :].min(axis=1))
                    ls = np.array(db_invitro_matrix.iloc[test_index, :].idxmin(axis=1))
                    conc = db_invitro["ec50"][ls]
                    label = db_invitro["invitro_label"][ls]
                    test_rf["invitro_conc"] = np.array(conc)
                    test_rf["invitro_label"] = np.array(label)
                    test_rf["invitro_dist"] = dist

        if encoding == "binary":

            df_score = fit_and_predict(
                model, train_rf, y_train, test_rf, y_test, encoding,
            )
            list_df_output.append(df_score)
        elif encoding == "multiclass":

            train_rf.loc[:, "target"] = y_train.values
            test_rf.loc[:, "target"] = y_test.values

            train_rf_h2o = h2o.H2OFrame(train_rf)
            test_rf_h2o = h2o.H2OFrame(test_rf)

            for col in train_rf.columns:
                if "label" in col:
                    train_rf_h2o[col] = train_rf_h2o[col].asfactor()
                    test_rf_h2o[col] = test_rf_h2o[col].asfactor()

            train_rf_h2o["target"] = train_rf_h2o["target"].asfactor()
            test_rf_h2o["target"] = test_rf_h2o["target"].asfactor()

            model.train(y="target", training_frame=train_rf_h2o)
            y_pred = model.predict(test_rf_h2o).as_data_frame()["predict"]
            df_score = pd.DataFrame()
            df_score.loc[0, "accuracy"] = accuracy_score(y_test, y_pred)
            df_score.loc[0, "recall"] = recall_score(y_test, y_pred, average="macro")
            df_score.loc[0, "specificity"] = np.nan
            df_score.loc[0, "f1"] = f1_score(y_test, y_pred, average="macro")
            df_score.loc[0, "precision"] = precision_score(
                y_test, y_pred, average="macro"
            )
            list_df_output.append(df_score)
        # print(train_rf.columns, train_rf.columns)
        del train_rf, test_rf

    df_output = pd.concat(list_df_output, axis=0)
    return df_output


def cal_matrixs_womor(X1, X2, categorical, non_categorical):
    matrices = {}
    basic_mat = euclidean_matrix(X1, X2, non_categorical)
    matrix_h = hamming_matrix(X1, X2, categorical)
    matrix_p = pubchem2d_matrix(X1, X2)
    matrices["euc"] = basic_mat
    matrices["hamming"] = matrix_h
    matrices["pubchem"] = matrix_p
    return matrices["hamming"], matrices["pubchem"]
=======
from helper_cte_model import *
from sklearn.model_selection import ParameterSampler
import h2o
import argparse
import random


def getArguments():
    parser = argparse.ArgumentParser(description="Running datafusion RASAR model for stratified splitting datasets, \
                                    without the input information from the mortality dataset but only information from \
                                     other effects or in vitro.")
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
        "-fixed", "--fixed_threshold", dest="fixed_threshold", default="no"
    )
    parser.add_argument("-ni", "--niter", dest="niter", default=20, type=int)
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

rand = random.randrange(1, 100)
test_size = 0.2
col_groups = "test_cas"


# -----------loading data & splitting into train and test dataset--------
print("loading dataset...", ctime())
db_mortality, db_datafusion = load_datafusion_datasets(
    args.inputFile,
    args.inputFile_df,
    categorical_columns=categorical,
    encoding=encoding,
    encoding_value=encoding_value,
)
print("finish loaded.", ctime())


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

Y_traintest = db_mortality.iloc[traintest_idx, :].conc1_mean.values
Y_valid = db_mortality.iloc[valid_idx, :].conc1_mean.values

print("calcultaing distance matrix..", ctime())
matrices = cal_matrixs(X_traintest, X_traintest, categorical, non_categorical)
matrices_df = cal_matrixs(
    X_traintest,
    db_datafusion.drop(columns="conc1_mean").copy(),
    categorical,
    non_categorical,
)
print("distance matrix successfully calculated!", ctime())

if encoding == "binary":
    model = RandomForestClassifier()
    hyper_params_tune = {
        "max_depth": [i for i in range(10, 100, 10)],
        "n_estimators": [int(x) for x in np.linspace(start=200, stop=1000, num=11)],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4, 8, 16, 32],
    }
elif encoding == "multiclass":
    h2o.init()
    h2o.no_progress()
    model = H2ORandomForestEstimator(seed=rand)
    hyper_params_tune = {
        "ntrees": [i for i in range(10, 1000, 10)],
        "max_depth": [i for i in range(10, 1000, 10)],
        "min_rows": [1, 10, 100, 1000],
        "sample_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "class_weight": [{0: i, 1: 1} for i in range(0, 20)],
    }


params_comb = list(
    ParameterSampler(hyper_params_tune, n_iter=args.niter, random_state=rand)
)

if args.alpha_h == "logspace":
    sequence_ap = np.logspace(-2, 0, 20)
    sequence_ah = sequence_ap
else:
    sequence_ap = [float(args.alpha_p)]
    sequence_ah = [float(args.alpha_h)]


best_accs = 0
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
            matrices_invitro="no",
            ah=ah,
            ap=ap,
            X=X_traintest,
            Y=Y_traintest,
            db_datafusion=db_datafusion,
            db_invitro="no",
            train_endpoint=args.train_endpoint,
            train_effect=args.train_effect,
            df_fishchem_tv=df_fishchem_tv,
            col_groups=col_groups,
            model=model,
            invitro="False",
            invitro_form="no",
            encoding=encoding,
            wmor=False,
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

    for k, v in params_comb[i].items():
        setattr(model, k, v)

    results = cv_datafusion_rasar(
        matrices,
        matrices_df,
        matrices_invitro="no",
        ah=best_ah,
        ap=best_ap,
        X=X_traintest,
        Y=Y_traintest,
        db_datafusion=db_datafusion,
        db_invitro="no",
        train_endpoint=args.train_endpoint,
        train_effect=args.train_effect,
        df_fishchem_tv=df_fishchem_tv,
        col_groups=col_groups,
        model=model,
        invitro="False",
        invitro_form="no",
        encoding=encoding,
        wmor=False,
    )
    if np.mean(results.accuracy) > best_accs:
        best_param = params_comb[i]
        best_accs = np.mean(results.accuracy)
        best_result = results
        print("success.", best_accs, end="\r")


df_mean = pd.DataFrame(best_result.mean(axis=0)).transpose()
df_std = pd.DataFrame(best_result.sem(axis=0)).transpose()


# -------------------tested on test dataset--------------------
print("start testing...", ctime())
for k, v in best_param.items():
    setattr(model, k, v)

traintest_index = X_traintest.index
valid_index = X_valid.index


X = db_mortality.drop(columns="conc1_mean").copy()

minmax = MinMaxScaler().fit(X_traintest[non_categorical])
X_traintest[non_categorical] = minmax.transform(X_traintest.loc[:, non_categorical])
X_valid[non_categorical] = minmax.transform(X_valid.loc[:, non_categorical])
X[non_categorical] = minmax.transform(X.loc[:, non_categorical])
db_datafusion[non_categorical] = minmax.transform(db_datafusion.loc[:, non_categorical])

db_datafusion_matrix = dist_matrix(
    X,
    db_datafusion.drop(columns="conc1_mean").copy(),
    non_categorical,
    categorical,
    best_ah,
    best_ap,
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

traintest_rf = df_rasar_traintest
test_rf = df_rasar_valid

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
        model, traintest_rf, Y_traintest, test_rf, Y_valid, encoding,
    )
elif encoding == "multiclass":

    traintest_rf.loc[:, "target"] = Y_traintest.values
    test_rf.loc[:, "target"] = Y_valid.values

    train_rf_h2o = h2o.H2OFrame(traintest_rf)
    test_rf_h2o = h2o.H2OFrame(test_rf)

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


# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv      -label ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o effect/R1_rasar_alphas/binary/df_rasar_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ACC.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o effect/R1_rasar_alphas/binary/df_rasar_ACC_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BCM.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o effect/R1_rasar_alphas/binary/df_rasar_BCM_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BEH.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o effect/R1_rasar_alphas/binary/df_rasar_BEH_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_CEL.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o effect/R1_rasar_alphas/binary/df_rasar_CEL_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ENZ.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o effect/R1_rasar_alphas/binary/df_rasar_ENZ_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_GEN.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o effect/R1_rasar_alphas/binary/df_rasar_GEN_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ITX.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o effect/R1_rasar_alphas/binary/df_rasar_ITX_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_PHY.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.0695193 -ap 0.78476 -o effect/R1_rasar_alphas/binary/df_rasar_PHY_noMOR.txt


# NEW
# python RASAR_simple.py   -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv   -e "multiclass" -ah 0.01 -ap 4.832930238571752  -o rasar/R1_knn_alphas/multiclass/s_rasar_mul.txt
# python RASAR_df_noMor.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv      -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_noMOR_mul.txt
# python RASAR_df_noMor.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ACC.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_ACC_noMOR_mul.txt
# python RASAR_df_noMor.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BCM.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_BCM_noMOR_mul.txt
# python RASAR_df_noMor.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BEH.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_BEH_noMOR_mul.txt
# python RASAR_df_noMor.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_CEL.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_CEL_noMOR_mul.txt
# python RASAR_df_noMor.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ENZ.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_ENZ_noMOR_mul.txt
# python RASAR_df_noMor.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_GEN.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_GEN_noMOR_mul.txt
# python RASAR_df_noMor.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ITX.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_ITX_noMOR_mul.txt
# python RASAR_df_noMor.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_PHY.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_PHY_noMOR_mul.txt
# python RASAR_df.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv      -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_mul.txt
# python RASAR_df.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ACC.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_ACC_mul.txt
# python RASAR_df.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BCM.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_BCM_mul.txt
# python RASAR_df.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BEH.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_BEH_mul.txt
# python RASAR_df.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_CEL.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_CEL_mul.txt
# python RASAR_df.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ENZ.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_ENZ_mul.txt
# python RASAR_df.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_GEN.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_GEN_mul.txt
# python RASAR_df.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ITX.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_ITX_mul.txt
# python RASAR_df.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_PHY.csv  -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_PHY_mul.txt


# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv      -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_noMOR.txt
# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ACC.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_ACC_noMOR.txt
# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BCM.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_BCM_noMOR.txt
# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BEH.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_BEH_noMOR.txt
# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_CEL.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_CEL_noMOR.txt
# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ENZ.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_ENZ_noMOR.txt
# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_GEN.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_GEN_noMOR.txt
# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ITX.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_ITX_noMOR.txt
# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_PHY.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_PHY_noMOR.txt
# python RASAR_df.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv      -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar.txt
# python RASAR_df.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ACC.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_ACC.txt
# python RASAR_df.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BCM.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_BCM.txt
# python RASAR_df.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BEH.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_BEH.txt
# python RASAR_df.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_CEL.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_CEL.txt
# python RASAR_df.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ENZ.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_ENZ.txt
# python RASAR_df.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_GEN.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_GEN.txt
# python RASAR_df.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ITX.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_ITX.txt
# python RASAR_df.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_PHY.csv  -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_PHY.txt
#
# knn binary
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv      -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ACC.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_ACC.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BCM.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_BCM.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BEH.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_BEH.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_CEL.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_CEL.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ENZ.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_ENZ.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_GEN.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_GEN.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ITX.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_ITX.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_PHY.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_PHY.txt

# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv      -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ACC.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_ACC_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BCM.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_BCM_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BEH.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_BEH_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_CEL.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_CEL_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ENZ.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_ENZ_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_GEN.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_GEN_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ITX.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_ITX_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_PHY.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.02592943797404667 -ap 10.0 -o rasar/R1_knn_alphas/binary/df_rasar_PHY_noMOR.txt


# rasar binary
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv      -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ACC.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_ACC.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BCM.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_BCM.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BEH.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_BEH.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_CEL.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_CEL.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ENZ.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_ENZ.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_GEN.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_GEN.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ITX.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_ITX.txt
# python RASAR_df_cte.py       -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_PHY.csv  -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_PHY.txt

# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv      -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ACC.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_ACC_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BCM.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_BCM_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_BEH.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_BEH_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_CEL.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_CEL_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ENZ.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_ENZ_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_GEN.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_GEN_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_ITX.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_ITX_noMOR.txt
# python RASAR_df_noMor_cte.py -i1 /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed_PHY.csv    -label ['LC50','EC50'] -effect 'MOR'  -ah 0.29763514416313175 -ap 1.0 -o rasar/R1_rasar_alphas/binary/df_rasar_PHY_noMOR.txt


# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv      -e "multiclass" -label ['LC50','EC50'] -effect 'MOR'  -ah 0.01 -ap 4.832930238571752 -o rasar/R1_knn_alphas/multiclass/df_rasar_noMOR_mul.txt
# python RASAR_df_noMor_cte.py -i1  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv      -label ['LC50','EC50'] -effect 'MOR'   -ah 0.01 -ap 0.0545559478116852 -o rasar/R1_s_alphas/df_rasar_noMOR.txt


def cv_datafusion_rasar_womor(
    matrices_df,
    db_invitro_matrix,
    ah,
    ap,
    X,
    Y,
    db_datafusion,
    db_invitro,
    train_endpoint,
    train_effect,
    df_fishchem_tv,
    col_groups,
    model=RandomForestClassifier(),
    invitro=False,
    invitro_form="both",
    encoding="binary",
):
    group_kfold = GroupKFold(n_splits=5)

    list_df_output = []
    for train_index, test_index in group_kfold.split(
        df_fishchem_tv, groups=df_fishchem_tv[col_groups]
    ):
        x_train = X.iloc[train_index]
        x_test = X.iloc[test_index]

        minmax = MinMaxScaler().fit(x_train[non_categorical])
        x_train[non_categorical] = minmax.transform(x_train.loc[:, non_categorical])
        x_test[non_categorical] = minmax.transform(x_test.loc[:, non_categorical])

        X_normed = X.copy()
        X_normed[non_categorical] = minmax.transform(X_normed.loc[:, non_categorical])

        db_datafusion_normed = db_datafusion.copy()
        db_datafusion_normed[non_categorical] = minmax.transform(
            db_datafusion_normed.loc[:, non_categorical]
        )

        db_datafusion_matrix = pd.DataFrame(
            ah * matrices_df["hamming"]
            + ap * matrices_df["pubchem"]
            + euclidean_matrix(X_normed, db_datafusion_normed, non_categorical,)
        )

        y_train = Y.iloc[train_index]
        y_test = Y.iloc[test_index]

        df_rasar_train, df_rasar_test = cal_df_rasar(
            train_index,
            test_index,
            X_normed.iloc[train_index],
            X_normed.iloc[test_index],
            db_datafusion_normed,
            db_datafusion_matrix,
            train_endpoint,
            train_effect,
            encoding,
        )

        train_rf = df_rasar_train
        test_rf = df_rasar_test

        if invitro == "own":
            train_rf = pd.DataFrame()
            test_rf = pd.DataFrame()

        if invitro != "False":
            if str(db_invitro) == "overlap":
                train_rf = get_vitroinfo(train_rf, X_normed, train_index, invitro_form)
                test_rf = get_vitroinfo(test_rf, X_normed, test_index, invitro_form)

                if (invitro != "False") & (invitro_form == "number"):
                    train_rf["invitro_conc"] = X_normed.ec50[train_index].reset_index(
                        drop=True
                    )
                    test_rf["invitro_conc"] = X_normed.ec50[test_index].reset_index(
                        drop=True
                    )

                elif (invitro != "False") & (invitro_form == "label"):
                    train_rf["invitro_label"] = X_normed.invitro_label[
                        train_index
                    ].reset_index(drop=True)
                    test_rf["invitro_label"] = X_normed.invitro_label[
                        test_index
                    ].reset_index(drop=True)

                elif (invitro != "False") & (invitro_form == "both"):
                    train_rf["ec50"] = X_normed.ec50[train_index].reset_index(drop=True)
                    test_rf["ec50"] = X_normed.ec50[test_index].reset_index(drop=True)
                    train_rf["invitro_label"] = X_normed.invitro_label[
                        train_index
                    ].reset_index(drop=True)
                    test_rf["invitro_label"] = X_normed.invitro_label[
                        test_index
                    ].reset_index(drop=True)
            else:
                if (invitro != "False") & (invitro_form == "number"):
                    dist = np.array(db_invitro_matrix.iloc[train_index, :].min(axis=1))
                    ls = np.array(db_invitro_matrix.iloc[train_index, :].idxmin(axis=1))
                    conc = db_invitro["ec50"][ls]
                    dist = db_invitro_matrix.lookup(
                        pd.Series(ls).index, pd.Series(ls).values
                    )
                    train_rf["invitro_conc"] = np.array(conc)
                    train_rf["invitro_dist"] = dist

                    dist = np.array(db_invitro_matrix.iloc[test_index, :].min(axis=1))
                    ls = np.array(db_invitro_matrix.iloc[test_index, :].idxmin(axis=1))
                    conc = db_invitro["ec50"][ls]
                    dist = db_invitro_matrix.lookup(
                        pd.Series(ls).index, pd.Series(ls).values
                    )
                    test_rf["invitro_conc"] = np.array(conc)
                    test_rf["invitro_dist"] = dist

                elif (invitro != "False") & (invitro_form == "label"):
                    dist = np.array(db_invitro_matrix.iloc[train_index, :].min(axis=1))
                    ls = np.array(db_invitro_matrix.iloc[train_index, :].idxmin(axis=1))
                    label = db_invitro["invitro_label"][ls]
                    dist = db_invitro_matrix.lookup(
                        pd.Series(ls).index, pd.Series(ls).values
                    )
                    train_rf["invitro_label"] = np.array(label)
                    train_rf["invitro_dist"] = dist

                    dist = np.array(db_invitro_matrix.iloc[test_index, :].min(axis=1))
                    ls = np.array(db_invitro_matrix.iloc[test_index, :].idxmin(axis=1))
                    label = db_invitro["invitro_label"][ls]
                    dist = db_invitro_matrix.lookup(
                        pd.Series(ls).index, pd.Series(ls).values
                    )
                    test_rf["invitro_label"] = np.array(label)
                    test_rf["invitro_dist"] = dist

                elif (invitro != "False") & (invitro_form == "both"):

                    dist = np.array(db_invitro_matrix.iloc[train_index, :].min(axis=1))
                    ls = np.array(db_invitro_matrix.iloc[train_index, :].idxmin(axis=1))
                    conc = db_invitro["ec50"][ls]
                    label = db_invitro["invitro_label"][ls]
                    train_rf["invitro_conc"] = np.array(conc)
                    train_rf["invitro_label"] = np.array(label)
                    train_rf["invitro_dist"] = dist

                    dist = np.array(db_invitro_matrix.iloc[test_index, :].min(axis=1))
                    ls = np.array(db_invitro_matrix.iloc[test_index, :].idxmin(axis=1))
                    conc = db_invitro["ec50"][ls]
                    label = db_invitro["invitro_label"][ls]
                    test_rf["invitro_conc"] = np.array(conc)
                    test_rf["invitro_label"] = np.array(label)
                    test_rf["invitro_dist"] = dist

        if encoding == "binary":

            df_score = fit_and_predict(
                model, train_rf, y_train, test_rf, y_test, encoding,
            )
            list_df_output.append(df_score)
        elif encoding == "multiclass":

            train_rf.loc[:, "target"] = y_train.values
            test_rf.loc[:, "target"] = y_test.values

            train_rf_h2o = h2o.H2OFrame(train_rf)
            test_rf_h2o = h2o.H2OFrame(test_rf)

            for col in train_rf.columns:
                if "label" in col:
                    train_rf_h2o[col] = train_rf_h2o[col].asfactor()
                    test_rf_h2o[col] = test_rf_h2o[col].asfactor()

            train_rf_h2o["target"] = train_rf_h2o["target"].asfactor()
            test_rf_h2o["target"] = test_rf_h2o["target"].asfactor()

            model.train(y="target", training_frame=train_rf_h2o)
            y_pred = model.predict(test_rf_h2o).as_data_frame()["predict"]
            df_score = pd.DataFrame()
            df_score.loc[0, "accuracy"] = accuracy_score(y_test, y_pred)
            df_score.loc[0, "recall"] = recall_score(y_test, y_pred, average="macro")
            df_score.loc[0, "specificity"] = np.nan
            df_score.loc[0, "f1"] = f1_score(y_test, y_pred, average="macro")
            df_score.loc[0, "precision"] = precision_score(
                y_test, y_pred, average="macro"
            )
            list_df_output.append(df_score)
        # print(train_rf.columns, train_rf.columns)
        del train_rf, test_rf

    df_output = pd.concat(list_df_output, axis=0)
    return df_output


def cal_matrixs_womor(X1, X2, categorical, non_categorical):
    matrices = {}
    basic_mat = euclidean_matrix(X1, X2, non_categorical)
    matrix_h = hamming_matrix(X1, X2, categorical)
    matrix_p = pubchem2d_matrix(X1, X2)
    matrices["euc"] = basic_mat
    matrices["hamming"] = matrix_h
    matrices["pubchem"] = matrix_p
    return matrices["hamming"], matrices["pubchem"]
>>>>>>> fb10ba16ae910083681208ee7642a81e312402b3:RASAR_df_noMor_cte.py
