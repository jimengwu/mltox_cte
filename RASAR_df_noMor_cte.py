from helper_model import *
from sklearn.model_selection import train_test_split, ParameterSampler
import h2o
from tqdm import tqdm
import argparse
import sys
import os
import random


def getArguments():
    parser = argparse.ArgumentParser(description="Running KNN_model for datasets.")
    parser.add_argument("-i1", "--input", dest="inputFile", required=True)
    parser.add_argument("-idf", "--input_df", dest="inputFile_df", required=True)
    parser.add_argument("-e", "--encoding", dest="encoding", default="binary")
    parser.add_argument(
        "-ah", "--alpha_h", dest="alpha_h", required=True, nargs="?", type=float
    )
    parser.add_argument(
        "-n", "--n_neighbors", dest="n_neighbors", nargs="?", default=1, type=int
    )
    parser.add_argument(
        "-ap", "--alpha_p", dest="alpha_p", required=True, nargs="?", type=float
    )
    parser.add_argument("-label", "--train_label", dest="train_label", required=True)
    parser.add_argument(
        "-fixed", "--fixed_threshold", dest="fixed_threshold", default="no"
    )
    parser.add_argument("-effect", "--train_effect", dest="train_effect", default="MOR")
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


def cv_datafusion_rasar(
    matrix_h_df,
    matrix_p_df,
    db_invitro_matrix,
    ah,
    ap,
    X,
    Y,
    db_datafusion,
    db_invitro,
    train_label,
    train_effect,
    model=RandomForestClassifier(),
    n_neighbors=1,
    invitro=False,
    invitro_form="both",
    encoding="binary",
):
    group_kfold = GroupKFold(n_splits=5)
    group_kfold.get_n_splits()

    accs = []
    sens = []
    precs = []
    specs = []
    f1 = []
    for train_index, test_index in group_kfold.split(
        df_fishchem_tv, groups=df_fishchem_tv[col_groups]
    ):
        x_train = X.iloc[train_index]
        x_test = X.iloc[test_index]
        minmax = MinMaxScaler().fit(x_train[non_categorical])
        x_train[non_categorical] = minmax.transform(x_train.loc[:, non_categorical])
        x_test[non_categorical] = minmax.transform(x_test.loc[:, non_categorical])
        new_X = X.copy()
        new_X[non_categorical] = minmax.transform(new_X.loc[:, non_categorical])
        new_db_datafusion = db_datafusion.copy()
        new_db_datafusion[non_categorical] = minmax.transform(
            new_db_datafusion.loc[:, non_categorical]
        )

        db_datafusion_matrix = pd.DataFrame(
            ah * matrix_h_df
            + ap * matrix_p_df
            + euclidean_matrix(
                new_X,
                new_db_datafusion.drop(columns="conc1_mean").copy(),
                non_categorical,
            )
        )

        y_train = Y.iloc[train_index]
        y_test = Y.iloc[test_index]

        datafusion_rasar_train, datafusion_rasar_test = cal_data_datafusion_rasar(
            train_index,
            test_index,
            new_X.iloc[train_index],
            new_X.iloc[test_index],
            new_db_datafusion,
            db_datafusion_matrix,
            train_label,
            train_effect,
            encoding,
        )

        # train_rf = pd.concat([simple_rasar_train, datafusion_rasar_train], axis=1)
        # test_rf = pd.concat([simple_rasar_test, datafusion_rasar_test], axis=1)

        train_rf = datafusion_rasar_train
        test_rf = datafusion_rasar_test
        if invitro == "own":
            train_rf = pd.DataFrame()
            test_rf = pd.DataFrame()
        if str(db_invitro) == "overlap":
            if (invitro != "False") & (invitro_form == "number"):
                train_rf["invitro_conc"] = new_X.ec50[train_index].reset_index(
                    drop=True
                )
                test_rf["invitro_conc"] = new_X.ec50[test_index].reset_index(drop=True)

            elif (invitro != "False") & (invitro_form == "label"):
                train_rf["invitro_label"] = new_X.invitro_label[
                    train_index
                ].reset_index(drop=True)
                test_rf["invitro_label"] = new_X.invitro_label[test_index].reset_index(
                    drop=True
                )

            elif (invitro != "False") & (invitro_form == "both"):
                train_rf["ec50"] = new_X.ec50[train_index].reset_index(drop=True)
                test_rf["ec50"] = new_X.ec50[test_index].reset_index(drop=True)
                train_rf["invitro_label"] = new_X.invitro_label[
                    train_index
                ].reset_index(drop=True)
                test_rf["invitro_label"] = new_X.invitro_label[test_index].reset_index(
                    drop=True
                )
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

            model.fit(train_rf, y_train)
            y_pred = model.predict(test_rf)

            accs.append(accuracy_score(y_test, y_pred))
            sens.append(recall_score(y_test, y_pred))
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            specs.append(tn / (tn + fp))
            precs.append(
                precision_score(
                    y_test,
                    y_pred,
                )
            )
            f1.append(f1_score(y_test, y_pred))
        elif encoding == "multiclass":

            train_rf.loc[:, "target"] = y_train
            test_rf.loc[:, "target"] = y_test

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

            # tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

            accs.append(accuracy_score(y_test, y_pred))
            sens.append(recall_score(y_test, y_pred, average="macro"))
            specs.append(np.nan)
            precs.append(precision_score(y_test, y_pred, average="macro"))
            f1.append(f1_score(y_test, y_pred, average="macro"))
        # print(train_rf.columns, train_rf.columns)
        del train_rf, test_rf

    results = {}
    results["avg_accs"] = np.mean(accs)
    results["se_accs"] = sem(accs)
    results["avg_sens"] = np.mean(sens)
    results["se_sens"] = sem(sens)
    results["avg_specs"] = np.mean(specs)
    results["se_specs"] = sem(specs)
    results["avg_precs"] = np.mean(precs)
    results["se_precs"] = sem(precs)
    results["avg_f1"] = np.mean(f1)
    results["se_f1"] = sem(f1)
    results["model"] = model
    results["ah"] = ah
    results["ap"] = ap

    return results


db_mortality, db_datafusion = load_datafusion_datasets(
    args.inputFile,
    args.inputFile_df,
    categorical_columns=categorical,
    encoding=encoding,
    encoding_value=encoding_value,
)
# db_mortality = db_mortality[:100]
# db_datafusion = db_datafusion[:300]


df_fishchem = db_mortality[["fish", "test_cas"]]
test_size = 0.2
col_groups = "test_cas"

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
if "conc1_mean" in list(X_trainvalid.columns):
    print("yes")
else:
    print("no")
Y_train = db_mortality.iloc[trainvalid_idx, :].conc1_mean
Y_valid = db_mortality.iloc[valid_idx, :].conc1_mean


print(ctime())


matrix_h_df, matrix_p_df = cal_matrixs(
    X_trainvalid,
    db_datafusion.drop(columns="conc1_mean").copy(),
    categorical,
    non_categorical,
)


print("distance matrix successfully calculated!", ctime())
# del db_mortality
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


params_comb = list(ParameterSampler(hyper_params_tune, n_iter=200, random_state=rand))

best_accs = 0
best_p = dict()

for i in tqdm(range(0, len(params_comb))):
    try:
        for k, v in params_comb[i].items():
            setattr(model, k, v)

        results = cv_datafusion_rasar(
            matrix_h_df,
            matrix_p_df,
            db_invitro_matrix="no",
            ah=args.alpha_h,
            ap=args.alpha_p,
            X=X_trainvalid,
            Y=Y_train,
            db_datafusion=db_datafusion,
            db_invitro="no",
            train_label=args.train_label,
            train_effect=args.train_effect,
            model=model,
            n_neighbors=args.n_neighbors,
            invitro="False",
            invitro_form="no",
            encoding=encoding,
        )
        if results["avg_accs"] > best_accs:
            best_p = params_comb[i]
            best_accs = results["avg_accs"]
            best_result = results
    except:
        continue


# -------------------tested on test dataset--------------------
print("start testing...", ctime())
for k, v in best_p.items():
    setattr(model, k, v)

train_index = X_trainvalid.index
test_index = X_valid.index


X = db_mortality.drop(columns="conc1_mean").copy()

minmax = MinMaxScaler().fit(X_trainvalid[non_categorical])
X_trainvalid[non_categorical] = minmax.transform(X_trainvalid.loc[:, non_categorical])
X_valid[non_categorical] = minmax.transform(X_valid.loc[:, non_categorical])
X[non_categorical] = minmax.transform(X.loc[:, non_categorical])
db_datafusion[non_categorical] = minmax.transform(db_datafusion.loc[:, non_categorical])

db_datafusion_matrix = dist_matrix(
    X,
    db_datafusion.drop(columns="conc1_mean").copy(),
    non_categorical,
    categorical,
    best_result["ah"],
    best_result["ap"],
)

datafusion_rasar_train, datafusion_rasar_test = cal_data_datafusion_rasar(
    train_index,
    test_index,
    X_trainvalid,
    X_valid,
    db_datafusion,
    db_datafusion_matrix,
    args.train_label,
    args.train_effect,
    encoding,
)
# del(matrix, db_datafusion_matrix)
# train_rf = pd.concat([simple_rasar_train, datafusion_rasar_train], axis=1)
# test_rf = pd.concat([simple_rasar_test, datafusion_rasar_test], axis=1)
train_rf = datafusion_rasar_train
test_rf = datafusion_rasar_test
print(train_rf.columns)
# del(datafusion_rasar_test, datafusion_rasar_train, simple_rasar_test, simple_rasar_train)
if encoding == "binary":

    model.fit(train_rf, Y_train)
    y_pred = model.predict(test_rf)

    accs = accuracy_score(Y_valid, y_pred)
    sens = recall_score(Y_valid, y_pred, average="macro")
    tn, fp, fn, tp = confusion_matrix(Y_valid, y_pred, labels=[0, 1]).ravel()
    specs = tn / (tn + fp)
    precs = precision_score(Y_valid, y_pred, average="macro")
    f1 = f1_score(Y_valid, y_pred, average="macro")
elif encoding == "multiclass":

    train_rf.loc[:, "target"] = Y_train
    test_rf.loc[:, "target"] = Y_valid

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
        best_result["se_accs"],
        sens,
        best_result["se_sens"],
        specs,
        best_result["se_specs"],
        precs,
        best_result["se_precs"],
        f1,
        best_result["se_f1"],
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
        best_result["se_accs"],
        sens,
        best_result["se_sens"],
        specs,
        best_result["se_specs"],
        precs,
        best_result["se_precs"],
        f1,
        best_result["se_f1"],
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
