from helper_model_cte import *
from sklearn.model_selection import train_test_split, ParameterSampler
import h2o
from tqdm import tqdm
import argparse
import sys
import os

class_threshold = None


def getArguments():
    parser = argparse.ArgumentParser(description="Running KNN_model for datasets.")
    parser.add_argument("-i", "--input", dest="inputFile", required=True)
    parser.add_argument("-i2", "--input_df", dest="inputFile_df", required=True)
    # parser.add_argument("-i3",
    #                     "--input_invitro",
    #                     dest="inputFile_invitro",
    #                     required=True)
    parser.add_argument(
        "-il", "--invitro_label", dest="invitro_label", default="number"
    )
    parser.add_argument("-dbi", "--db_invitro", dest="db_invitro", default="noinvitro")
    parser.add_argument("-wi", "--w_invitro", dest="w_invitro", default="False")
    parser.add_argument("-ni", "--niter", dest="niter", default=50, type=int)
    parser.add_argument("-e", "--encoding", dest="encoding", default="binary")
    parser.add_argument("-ah", "--alpha_h", dest="alpha_h", required=True)
    parser.add_argument("-ap", "--alpha_p", dest="alpha_p", required=True)
    parser.add_argument(
        "-n", "--n_neighbors", dest="n_neighbors", nargs="?", default=1, type=int
    )
    parser.add_argument(
        "-endpoint", "--train_endpoint", dest="train_endpoint", required=True
    )
    # parser.add_argument("-fixed", "--fixed_threshold", dest="fixed_threshold")
    parser.add_argument("-effect", "--train_effect", dest="train_effect", required=True)
    parser.add_argument("-o", "--output", dest="outputFile", default="binary.txt")
    return parser.parse_args()


args = getArguments()
if args.encoding == "binary":
    encoding = "binary"
    encoding_value = 1
elif args.encoding == "multiclass":
    encoding = "multiclass"
    encoding_value = [0.1, 1, 10, 100]

if args.db_invitro != "noinvitro":
    db_mortality, db_datafusion, db_invitro = load_datafusion_datasets_invitro(
        args.inputFile,
        args.inputFile_df,
        args.db_invitro,
        categorical_columns=categorical,
        non_categorical_columns=non_categorical,
        encoding=encoding,
        encoding_value=encoding_value,
    )
else:
    db_mortality, db_datafusion = load_datafusion_datasets(
        args.inputFile,
        args.inputFile_df,
        categorical_columns=categorical,
        encoding=encoding,
        encoding_value=encoding_value,
    )
    db_invitro = args.db_invitro

# db_mortality=db_mortality[:200]
# db_datafusion=db_datafusion[:500]
test_size = 0.2
col_groups = "test_cas"

df_fishchem = db_mortality[["fish", "test_cas"]]
trainvalid_idx, valid_idx = get_grouped_train_test_split(
    df_fishchem, test_size, col_groups
)
df_fishchem_tv = df_fishchem.iloc[trainvalid_idx, :]


X = db_mortality.drop(columns="conc1_mean").copy()
Y = db_mortality.conc1_mean.values

# X_trainvalid, X_valid, Y_trainvalid, Y_valid = train_test_split(
#     X, Y, test_size=0.2, random_state=42
# )


X_trainvalid = X.iloc[trainvalid_idx, :]
X_valid = X.iloc[valid_idx, :]
Y_trainvalid = Y[trainvalid_idx]
Y_valid = Y[valid_idx]


print("Data loaded.", ctime())
matrix_euc, matrix_h, matrix_p = cal_matrixs(
    X_trainvalid, X_trainvalid, categorical, non_categorical
)
matrix_euc_df, matrix_h_df, matrix_p_df = cal_matrixs(
    X_trainvalid, db_datafusion, categorical, non_categorical
)

matrix_euc_x_invitro, matrix_h_x_invitro, matrix_p_x_invitro = cal_matrixs(
    X_trainvalid, db_invitro, categorical_both, non_categorical
)

print("distance matrix successfully calculated!", ctime())
del db_mortality
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
params_comb = list(ParameterSampler(hyper_params_tune, n_iter=args.n_iter))


if args.alpha_h == "logspace":
    sequence_ap = np.logspace(-2, 0, 20)
    sequence_ah = sequence_ap
else:
    sequence_ap = [float(args.alpha_p)]
    sequence_ah = [float(args.alpha_h)]

best_accs = 0
best_p = dict()
count = 1
for ah in sequence_ah:
    for ap in sequence_ap:
        for i in range(0, len(params_comb)):
            # try:
            for k, v in params_comb[i].items():
                setattr(model, k, v)
            print(
                ah,
                ap,
                "*" * 50,
                count / (len(sequence_ap) ** 2 * len(params_comb)),
                ctime(),
                end="\r",
            )
            results = cv_datafusion_rasar_new(
                matrix_euc,
                matrix_h,
                matrix_p,
                matrix_euc_df,
                matrix_h_df,
                matrix_p_df,
                db_invitro_matrix=(
                    matrix_h_x_invitro,
                    matrix_p_x_invitro,
                    matrix_euc_x_invitro,
                ),
                ah=ah,
                ap=ap,
                X=X_trainvalid,
                Y=Y_trainvalid,
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
                best_p = params_comb[i]
                best_accs = np.mean(results.accuracy)
                best_result = results
                best_ah = ah
                best_ap = ap
                print("success.", best_accs)
            count = count + 1
            # except:
            #     continue

df_mean = pd.DataFrame(best_result.mean(axis=0)).transpose()
df_std = pd.DataFrame(best_result.sem(axis=0)).transpose()


# -------------------tested on test dataset--------------------
print("start testing...", ctime())
for k, v in best_p.items():
    setattr(model, k, v)

train_index = X_trainvalid.index
test_index = X_valid.index

if args.w_invitro == "own":
    train_rf = pd.DataFrame()
    test_rf = pd.DataFrame()
else:
    minmax = MinMaxScaler().fit(X_trainvalid[non_categorical])
    X_trainvalid[non_categorical] = minmax.transform(
        X_trainvalid.loc[:, non_categorical]
    )
    X_valid[non_categorical] = minmax.transform(X_valid.loc[:, non_categorical])
    X[non_categorical] = minmax.transform(X.loc[:, non_categorical])
    db_invitro[non_categorical] = minmax.transform(db_invitro.loc[:, non_categorical])
    db_datafusion[non_categorical] = minmax.transform(
        db_datafusion.loc[:, non_categorical]
    )

    matrix_test = dist_matrix(
        X_valid, X_trainvalid, non_categorical, categorical, best_ah, best_ap
    )
    matrix_train = dist_matrix(
        X_trainvalid, X_trainvalid, non_categorical, categorical, best_ah, best_ap
    )

    db_invitro_matrix_train = dist_matrix(
        X_trainvalid, db_invitro, non_categorical, categorical_both, best_ah, best_ap
    )
    db_invitro_matrix_test = dist_matrix(
        X_valid, db_invitro, non_categorical, categorical_both, best_ah, best_ap
    )

    db_datafusion_matrix = dist_matrix(
        X, db_datafusion, non_categorical, categorical, best_ah, best_ap
    )

    del (matrix_euc, matrix_h, matrix_p, matrix_euc_df, matrix_h_df, matrix_p_df)

    simple_rasar_train, simple_rasar_test = cal_s_rasar(
        matrix_train, matrix_test, Y_trainvalid, args.n_neighbors, args.encoding
    )

    datafusion_rasar_train, datafusion_rasar_test = cal_df_rasar(
        train_index,
        test_index,
        X_trainvalid,
        X_valid,
        db_datafusion,
        db_datafusion_matrix,
        args.train_endpoint,
        args.train_effect,
        encoding,
    )
    # del(matrix, db_datafusion_matrix)
    train_rf = pd.concat([simple_rasar_train, datafusion_rasar_train], axis=1)
    test_rf = pd.concat([simple_rasar_test, datafusion_rasar_test], axis=1)


if args.w_invitro != "False":
    if str(args.db_invitro) == "overlap":
        train_rf = get_vitroinfo(
            train_rf, X_trainvalid, train_index, args.invitro_label
        )
        test_rf = get_vitroinfo(test_rf, X_valid, test_index, args.invitro_label)
    else:
        db_invitro_matrix = dist_matrix(
            X, db_invitro, non_categorical, categorical_both, best_ah, best_ap
        )
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
if encoding == "binary":
    df_test_score = fit_and_predict(
        model,
        train_rf,
        Y_trainvalid,
        test_rf,
        Y_valid,
        encoding,
    )

elif encoding == "multiclass":

    train_rf.loc[:, "target"] = Y_trainvalid
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

    df_test_score = pd.DataFrame()
    df_test_score.loc[0, "accuracy"] = accuracy_score(Y_valid, y_pred)
    df_test_score.loc[0, "recall"] = recall_score(Y_valid, y_pred, average="macro")
    df_test_score.loc[0, "specificity"] = np.nan
    df_test_score.loc[0, "f1"] = f1_score(Y_valid, y_pred, average="macro")
    df_test_score.loc[0, "precision"] = precision_score(
        Y_valid, y_pred, average="macro"
    )
    h2o.shutdown()


df_output = pd.concat(
    [df_mean, df_std, df_test_score],
    keys=["train_mean", "train_std", "test"],
    names=["series_name"],
)
df_output["model"] = str(best_p)

# ----------------save the information into a file-------
df2file(df_output, args.outputFile)


# ----------------------------------------------------------------------------------general: invitro + (invivo) -> invivo(gtvvdf)--------------------------------------------
# binary, R=4
# python RASAR_df_addinginvitro_cte.py -i /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -i2 /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv -dbi /local/wujimeng/code_jimeng/data/invitro/invitro_processed.csv  -endpoint ['LC50','EC50'] -effect 'MOR' -wi "True" -il "number" -n 4 -ah 0.23357214690901212 -ap 0.1128837891684889  -o "vitro_e/vivo+vitro/bestR/general/general_df.txt"
# python RASAR_df_addinginvitro_cte.py -i /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -i2 /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv -dbi /local/wujimeng/code_jimeng/data/invitro/invitro_processed.csv  -endpoint ['LC50','EC50'] -effect 'MOR' -wi "own"  -il "number" -n 4 -ah 0.23357214690901212 -ap 0.1128837891684889  -o "vitro_e/vivo+vitro/bestR/general/general_own_invitro_df.txt"

# multiclass, R=5
# python RASAR_df_addinginvitro.py -i data/invivo/lc50_processed_jim.csv  -i2 data/invivo/datafusion_db_processed.csv -dbi data/invitro/invitro_processed.csv -e "multiclass" -endpoint ['LC50','EC50'] -effect 'MOR' -wi "True" -il "number" -n 5 -ah 0.04281332398719394 -ap 0.7847599703514611  -o "vitro_e/vivo+vitro/bestR/new/multiclass/general/general_df_mul.txt"
# python RASAR_df_addinginvitro.py -i data/invivo/lc50_processed_jim.csv  -i2 data/invivo/datafusion_db_processed.csv -dbi data/invitro/invitro_processed.csv -e "multiclass" -endpoint ['LC50','EC50'] -effect 'MOR' -wi "own"  -il "number" -n 5 -ah 0.04281332398719394 -ap 0.7847599703514611  -o "vitro_e/vivo+vitro/bestR/new/multiclass/general/general_own_invitro_df_mul.txt"


# invitro_form = args.invitro_label
# invitro = args.w_invitro


# if invitro == "own":
#     train_rf = pd.DataFrame()
#     test_rf = pd.DataFrame()

# if str(db_invitro) == "overlap":
#     if (invitro != "False") & (invitro_form == "number"):
#         train_rf["invitro_conc"] = X_trainvalid.invitro_conc.reset_index(drop=True)
#         test_rf["invitro_conc"] = X_valid.invitro_conc.reset_index(drop=True)

#     elif (invitro != "False") & (invitro_form == "label"):
#         train_rf["invitro_label"] = X_trainvalid.invitro_label.reset_index(drop=True)
#         test_rf["invitro_label"] = X_valid.invitro_label.reset_index(drop=True)

#     elif (invitro != "False") & (invitro_form == "both"):
#         train_rf["invitro_conc"] = X_trainvalid.invitro_conc.reset_index(drop=True)
#         test_rf["invitro_conc"] = X_valid.invitro_conc.reset_index(drop=True)
#         train_rf["invitro_label"] = X_trainvalid.invitro_label.reset_index(drop=True)
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

# del (
#     datafusion_rasar_test,
#     datafusion_rasar_train,
#     simple_rasar_test,
#     simple_rasar_train,
# )
# print(train_rf.columns[:30])
# if encoding == "binary":

#     model.fit(train_rf, Y_trainvalid)
#     y_pred = model.predict(test_rf)

#     accs = accuracy_score(Y_valid, y_pred)
#     sens = recall_score(Y_valid, y_pred, average="macro")
#     tn, fp, fn, tp = confusion_matrix(Y_valid, y_pred, labels=[0, 1]).ravel()
#     specs = tn / (tn + fp)
#     precs = precision_score(Y_valid, y_pred, average="macro")
#     f1 = f1_score(Y_valid, y_pred, average="macro")
# elif encoding == "multiclass":

#     train_rf.loc[:, "target"] = Y_trainvalid
#     test_rf.loc[:, "target"] = Y_valid

#     train_rf_h2o = h2o.H2OFrame(train_rf)
#     test_rf_h2o = h2o.H2OFrame(test_rf)

#     for col in train_rf.columns:
#         if "label" in col:
#             train_rf_h2o[col] = train_rf_h2o[col].asfactor()
#             test_rf_h2o[col] = test_rf_h2o[col].asfactor()

#     train_rf_h2o["target"] = train_rf_h2o["target"].asfactor()
#     test_rf_h2o["target"] = test_rf_h2o["target"].asfactor()

#     model.train(y="target", training_frame=train_rf_h2o)
#     y_pred = model.predict(test_rf_h2o).as_data_frame()["predict"]

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
#     \nSensitivity:  {}, Se.Sensitivity: {}
#         \nSpecificity:  {}, Se.Specificity:{}
#     \nPrecision:  {}, Se.Precision: {}
#     \nf1_score:{}, Se.f1_score:{}""".format(
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
# info.append("Random state".format(rand))
# filename = args.outputFile
# dirname = os.path.dirname(filename)
# if not os.path.exists(dirname):
#     os.makedirs(dirname)

# with open(filename, "w") as file_handler:
#     for item in info:
#         file_handler.write("{}\n".format(item))


# ---------------------------------old:
# python RASAR_df_addinginvitro.py -i data/invivo/lc50_processed_jim.csv  -i2 data/invivo/datafusion_db_processed.csv -dbi data/invitro/invitro_merged.csv -endpoint ['LC50','EC50'] -effect 'MOR' -wi "True"  -ah 0.041753189365604 -ap 4.893900918477494  -o "vivo+vitro(t&e)/general_df.txt"
# python RASAR_df_addinginvitro.py -i data/invivo/lc50_processed_jim.csv  -i2 data/invivo/datafusion_db_processed.csv -dbi data/invitro/invitro_merged.csv -endpoint ['LC50','EC50'] -effect 'MOR' -wi "own"   -ah 0.041753189365604 -ap 4.893900918477494  -o "vivo+vitro(t&e)/general_own_invitro_df.txt"
# python RASAR_df_addinginvitro.py -i data/invivo/lc50_processed_jim.csv  -i2 data/invivo/datafusion_db_processed.csv -dbi data/invitro/invitro_merged.csv -endpoint ['LC50','EC50'] -effect 'MOR' -wi "False" -ah 0.041753189365604 -ap 4.893900918477494  -o invitro/general_compared_df.txt
#


# python RASAR_df_addinginvitro.py -i data/invivo/lc50_processed_jim.csv  -i2 data/invivo/datafusion_db_processed.csv -dbi data/invitro/invitro_merged.csv     -endpoint ['LC50','EC50'] -effect 'MOR' -wi "True" -il "number" -n 4 -ah 0.23357214690901212 -ap 0.1128837891684889  -o "vivo+vitro(t&e)/bestR/general_df.txt"
# python RASAR_df_addinginvitro.py -i data/invivo/lc50_processed_jim.csv  -i2 data/invivo/datafusion_db_processed.csv -dbi data/invitro/invitro_merged.csv     -endpoint ['LC50','EC50'] -effect 'MOR' -wi "own"  -il "number" -n 4 -ah 0.23357214690901212 -ap 0.1128837891684889  -o "vivo+vitro(t&e)/bestR/general_own_invitro_df.txt"
# python RASAR_df_addinginvitro.py -i data/invivo/lc50_processed_jim.csv  -i2 data/invivo/datafusion_db_processed.csv -dbi data/TOXCAST/toxcast_processed.csv  -endpoint ['LC50','EC50'] -effect 'MOR' -wi "True" -il "number" -n 4 -ah 0.23357214690901212 -ap 0.1128837891684889  -o "vitro_t/vivo&vitro/bestR/general_df.txt"
# python RASAR_df_addinginvitro.py -i data/invivo/lc50_processed_jim.csv  -i2 data/invivo/datafusion_db_processed.csv -dbi data/TOXCAST/toxcast_processed.csv  -endpoint ['LC50','EC50'] -effect 'MOR' -wi "own"  -il "number" -n 4 -ah 0.23357214690901212 -ap 0.1128837891684889  -o "vitro_t/vivo&vitro/bestR/general_own_invitro_df.txt"
# python RASAR_df_addinginvitro.py -i data/invivo/lc50_processed_jim.csv  -i2 data/invivo/datafusion_db_processed.csv -dbi data/invitro/invitro_merged.csv     -endpoint ['LC50','EC50'] -effect 'MOR' -wi "False"             -n 4 -ah 0.23357214690901212 -ap 0.1128837891684889  -o "vivo+vitro(t&e)/bestR/general_false_df.txt"


# python RASAR_df_addinginvitro.py -i data/invivo/lc50_processed_jim.csv  -i2 data/invivo/datafusion_db_processed.csv -dbi data/invitro/invitro_merged.csv     -endpoint ['LC50','EC50'] -effect 'MOR' -wi "True"                   -ah 0.041753189365604   -ap 4.893900918477494   -o "vivo+vitro(t&e)/general_df.txt"
# python RASAR_df_addinginvitro.py -i data/invivo/lc50_processed_jim.csv  -i2 data/invivo/datafusion_db_processed.csv -dbi data/invitro/invitro_merged.csv     -endpoint ['LC50','EC50'] -effect 'MOR' -wi "own"                    -ah 0.041753189365604   -ap 4.893900918477494   -o "vivo+vitro(t&e)/general_own_invitro_df.txt"
#
