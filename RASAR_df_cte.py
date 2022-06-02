from helper_model_cte import *
from sklearn.model_selection import train_test_split, ParameterSampler
import h2o
from tqdm import tqdm
import argparse
import sys
import os


def getArguments():
    parser = argparse.ArgumentParser(description="Running KNN_model for datasets.")
    parser.add_argument("-i1", "--input", dest="inputFile", required=True)
    parser.add_argument("-idf", "--input_df", dest="inputFile_df", required=True)
    parser.add_argument(
        "-il", "--invitro_label", dest="invitro_label", default="number"
    )
    parser.add_argument("-dbi", "--db_invitro", dest="db_invitro", default="noinvitro")
    parser.add_argument("-wi", "--w_invitro", dest="w_invitro", default="False")
    parser.add_argument("-e", "--encoding", dest="encoding", default="binary")
    parser.add_argument("-ah", "--alpha_h", dest="alpha_h", required=True, nargs="?")
    parser.add_argument("-ap", "--alpha_p", dest="alpha_p", required=True, nargs="?")
    parser.add_argument(
        "-n", "--n_neighbors", dest="n_neighbors", nargs="?", default=1, type=int
    )
    parser.add_argument(
        "-endpoint", "--train_endpoint", dest="train_endpoint", required=True
    )
    # parser.add_argument("-fixed", "--fixed_threshold", dest="fixed_threshold")
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

# print(args.outputFile)
rand = random.randrange(1, 100)

test_size = 0.2
col_groups = "test_cas"

db_mortality, db_datafusion = load_datafusion_datasets(
    args.inputFile,
    args.inputFile_df,
    categorical_columns=categorical,
    encoding=encoding,
    encoding_value=encoding_value,
)
# db_mortality = db_mortality[:300]
# db_datafusion = db_datafusion[:300]


df_fishchem = db_mortality[["fish", "test_cas"]]
trainvalid_idx, test_idx = get_grouped_train_test_split(
    df_fishchem, test_size, col_groups
)
df_fishchem_tv = df_fishchem.iloc[trainvalid_idx, :].reset_index(drop=True)

X = db_mortality.drop(columns="conc1_mean").copy()
X_trainvalid = db_mortality.drop(columns="conc1_mean").iloc[trainvalid_idx, :]
X_valid = db_mortality.drop(columns="conc1_mean").iloc[test_idx, :]
Y_trainvalid = db_mortality.iloc[trainvalid_idx, :].conc1_mean.values
Y_valid = db_mortality.iloc[test_idx, :].conc1_mean.values


print("Data loaded.", ctime())
# matrix_h, matrix_p = cal_matrixs(
#     X_trainvalid, X_trainvalid, categorical, non_categorical
# )
print("calcultaing distance matrix..", ctime())
matrix_euc, matrix_h, matrix_p = cal_matrixs(
    X_trainvalid, X_trainvalid, categorical, non_categorical
)
matrix_euc_df, matrix_h_df, matrix_p_df = cal_matrixs(
    X_trainvalid,
    db_datafusion.drop(columns="conc1_mean").copy(),
    categorical,
    non_categorical,
)

print("distance matrix successfully calculated!", ctime())

del db_mortality
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


params_comb = list(ParameterSampler(hyper_params_tune, n_iter=10, random_state=rand))


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
            print(
                "*" * 50,
                count / (len(sequence_ap) ** 2 * len(params_comb)),
                ctime(),
                end="\r",
            )
            try:
                for k, v in params_comb[i].items():
                    setattr(model, k, v)

                results = cv_datafusion_rasar_new(
                    matrix_euc,
                    matrix_h,
                    matrix_p,
                    matrix_euc_df,
                    matrix_h_df,
                    matrix_p_df,
                    db_invitro_matrix="no",
                    ah=ah,
                    ap=ap,
                    X=X_trainvalid,
                    Y=Y_trainvalid,
                    db_datafusion=db_datafusion,
                    db_invitro=args.db_invitro,
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
                    best_p = params_comb[i]
                    best_accs = np.mean(results.accuracy)
                    best_result = results
                    best_ah = ah
                    best_ap = ap
                    print("success.", best_accs)

            except:
                continue
            count = count + 1

df_mean = pd.DataFrame(best_result.mean(axis=0)).transpose()
df_std = pd.DataFrame(best_result.sem(axis=0)).transpose()

# -------------------tested on test dataset--------------------
print("start testing...", ctime())
for k, v in best_p.items():
    setattr(model, k, v)

train_index = X_trainvalid.index
test_index = X_valid.index

minmax = MinMaxScaler().fit(X_trainvalid[non_categorical])
X_trainvalid[non_categorical] = minmax.transform(X_trainvalid.loc[:, non_categorical])
X_valid[non_categorical] = minmax.transform(X_valid.loc[:, non_categorical])
X[non_categorical] = minmax.transform(X.loc[:, non_categorical])
db_datafusion[non_categorical] = minmax.transform(db_datafusion.loc[:, non_categorical])


matrix_test = dist_matrix(
    X_valid,
    X_trainvalid,
    non_categorical,
    categorical,
    best_ah,
    best_ap,
)
matrix_train = dist_matrix(
    X_trainvalid,
    X_trainvalid,
    non_categorical,
    categorical,
    best_ah,
    best_ap,
)

db_datafusion_matrix = dist_matrix(
    X,
    db_datafusion.drop(columns="conc1_mean").copy(),
    non_categorical,
    categorical,
    best_ah,
    best_ap,
)


del (matrix_h, matrix_p, matrix_h_df, matrix_p_df)

simple_rasar_train, simple_rasar_test = cal_s_rasar(
    matrix_train,
    matrix_test,
    Y_trainvalid,
    args.n_neighbors,
    encoding,
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

train_rf = pd.concat([simple_rasar_train, datafusion_rasar_train], axis=1)
test_rf = pd.concat([simple_rasar_test, datafusion_rasar_test], axis=1)

if args.w_invitro == "own":
    train_rf = pd.DataFrame()
    test_rf = pd.DataFrame()

if args.w_invitro != "False":
    if str(args.db_invitro) == "overlap":
        train_rf = get_vitroinfo(
            train_rf, X_trainvalid, train_index, args.invitro_label
        )
        test_rf = get_vitroinfo(test_rf, X_valid, test_index, args.invitro_label)

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


# python RASAR_df.py -i1 data/LOEC/loec_processed.csv  -i2 data/LOEC/loec_processed_df_itself.csv -endpoint 'LOEC' -effect 'MOR' -fixed no -ah 0.615848211066026 -ap 16.23776739188721 -o results/mortality/loec_df_itself.txt
# python RASAR_df.py -i1 data/NOEC/noec_processed.csv  -i2 data/NOEC/noec_processed_df_itself.csv -endpoint 'NOEC' -effect 'MOR' -fixed no -ah 0.06951927961775606  -ap 0.2069138081114788 -o results/mortality/noec_df_itself.txt
# python RASAR_df.py -i1 data/LC50/lc50_processed.csv  -i2 data/LC50/lc50_processed_df_itself.csv  -endpoint ['LC50','EC50'] -effect 'MOR' -fixed no -ah 0.2069138081114788 -ap 0.615848211066026 -o results/mortality/lc50_df_itself.txt
# python RASAR.py -i1 data/LC50/lc50_processed_rainbow.csv  -i2 data/LC50/lc50_processed_df_rainbow.csv -endpoint ['LC50','EC50'] -effect 'MOR' -fixed no -ah 5.455594781168514 -ap 143.8449888287663 -o results/rainbow/lc50_df_rainbow_binary.txt

# python RASAR.py  -i1 data/LC50/lc50_processed.csv  -idf data/LC50/lc50_processed_df_acc.csv  -endpoint ['LC50','EC50'] -effect 'MOR' -ah 0.2069138081114788 -ap 0.615848211066026 -o results/effect/lc50_df_acc_nomor.txt
# python RASAR.py  -i1 data/LC50/lc50_processed.csv  -idf data/LC50/lc50_processed_df_beh.csv  -endpoint ['LC50','EC50'] -effect 'MOR' -ah 0.2069138081114788 -ap 0.615848211066026 -o results/effect/lc50_df_beh_nomor.txt
# python RASAR.py  -i1 data/LC50/lc50_processed.csv  -i2 data/LC50/lc50_processed_df_enz.csv  -endpoint ['LC50','EC50'] -effect 'MOR' -ah 0.2069138081114788 -ap 0.615848211066026 -o results/effect/lc50_df_enz_nomor.txt
# python RASAR.py  -i1 data/LC50/lc50_processed.csv  -i2 data/LC50/lc50_processed_df_gen.csv  -endpoint ['LC50','EC50'] -effect 'MOR' -ah 0.2069138081114788 -ap 0.615848211066026 -o results/effect/lc50_df_gen_nomor.txt
# python RASAR.py  -i1 data/LC50/lc50_processed.csv  -i2 data/LC50/lc50_processed_df_bcm.csv  -endpoint ['LC50','EC50'] -effect 'MOR' -ah 0.2069138081114788 -ap 0.615848211066026 -o results/effect/lc50_df_bcm_nomor.txt

# python RASAR_df.py -i1 lc_db_processed.csv  -idf datafusion_db_processed.csv  -endpoint ['LC50','EC50'] -effect 'MOR' -fixed no -ah 0.5623 -ap 0.0749 -o rasar/df_bi.txt
# python RASAR_df.py -i1 lc_db_processed.csv  -idf datafusion_db_processed.csv  -endpoint ['LC50','EC50'] -effect 'MOR' -fixed no -ah 0.31622776601683794 -ap 3.625  -o rasar/df_bi2.txt
# python RASAR_df.py -i1 lc_db_processed.csv  -idf datafusion_db_processed.csv  -endpoint ['LC50','EC50'] -effect 'MOR' -fixed no -ah 1 -ap 1  -o rasar/df_table4_bi.txt

# python RASAR_df.py -i1 lc_db_processed.csv  -idf datafusion_db_processed.csv  -e "multiclass" -endpoint ['LC50','EC50'] -effect 'MOR' -fixed no -ah 0.01268961003167922 -ap 0.3562247890262442  -o rasar/df_mul_test.txt
# python RASAR_df.py -i1 lc_db_processed.csv  -idf datafusion_db_processed.csv  -e "multiclass" -endpoint ['LC50','EC50'] -effect 'MOR' -fixed no -ah 0.008531678524172814 -ap 0.3039195382313201  -o rasar/df_mul_test2.txt
#

# python RASAR_df.py -i1 lc_db_processed.csv  -idf datafusion_db_processed.csv  -e "multiclass" -endpoint ['LC50','EC50'] -effect 'MOR' -fixed no -ah 0.01268961003167922 -ap 0.3562247890262442  -o rasar/df_mul_new2.txt
#


# ------------------------------------------------------invitro to invitro(ottdf)----------------------------------------------------------
# python RASAR_df_cte.py -i1 /local/wujimeng/code_jimeng/data/invitro/invitro_eawag_repeated.csv  -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv -endpoint ['LC50','EC50'] -effect 'MOR'  -ah logspace -ap logspace -dbi "no"      -wi "False" -il "both"   -o vitro_e/vitro/vitro_df_rasar_bi_repeated_4.csv


# ------------------------------------------------------invitro to invitro(gtt)----------------------------------------------------------
# binary:
# python RASAR_df_cte.py     -i1 /local/wujimeng/code_jimeng/data/invitro/invitro_processed.csv   -idf  /local/wujimeng/code_jimeng/data/invivo/datafusion_db_processed.csv  -endpoint ['LC50','EC50'] -ah 0.23357214690901212 -ap 0.11288378916846889 -n 4 -o vitro_e/vivo+vitro/general/vitro_df_rasar_bi_4.csv

# multiclass & R=5:

# invitro_form = args.invitro_label
# db_invitro = args.db_invitro
# invitro = args.w_invitro
# if str(db_invitro) == "overlap":
#     if (invitro != "False") & (invitro_form == "number"):
#         train_rf["invitro_conc"] = X_trainvalid.invitro_conc.reset_index(drop=True)
#         test_rf["invitro_conc"] = X_valid.invitro_conc.reset_index(drop=True)

#     elif (invitro != "False") & (invitro_form == "label"):
#         train_rf["invitro_label"] = X_trainvalid.invitro_label_half.reset_index(
#             drop=True
#         )
#         test_rf["invitro_label"] = X_valid.invitro_label_half.reset_index(drop=True)

#     elif (invitro != "False") & (invitro_form == "both"):
#         train_rf["invitro_conc"] = X_trainvalid.invitro_conc.reset_index(drop=True)
#         test_rf["invitro_conc"] = X_valid.invitro_conc.reset_index(drop=True)
#         train_rf["invitro_label"] = X_trainvalid.invitro_label_half.reset_index(
#             drop=True
#         )
#         test_rf["invitro_label"] = X_valid.invitro_label_half.reset_index(drop=True)

#     elif (invitro != "False") & (invitro_form == "label_half"):
#         train_rf["invitro_label_half"] = X.iloc[
#             train_index, :
#         ].invitro_label.reset_index(drop=True)
#         test_rf["invitro_label_half"] = X.iloc[test_index, :].invitro_label.reset_index(
#             drop=True
#         )

#     elif (invitro != "False") & (invitro_form == "both_half"):
#         train_rf["invitro_conc"] = X.iloc[train_index, :].invitro_conc.reset_index(
#             drop=True
#         )
#         test_rf["invitro_conc"] = X.iloc[test_index, :].invitro_conc.reset_index(
#             drop=True
#         )
#         train_rf["invitro_label_half"] = X.iloc[
#             train_index, :
#         ].invitro_label.reset_index(drop=True)
#         test_rf["invitro_label_half"] = X.iloc[test_index, :].invitro_label.reset_index(
#             drop=True
#         )


# del (
#     datafusion_rasar_test,
#     datafusion_rasar_train,
#     simple_rasar_test,
#     simple_rasar_train,
# )
# print(train_rf.columns)
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
#         best_result["se_accs"],
#         sens,
#         best_result["se_sens"],
#         specs,
#         best_result["se_specs"],
#         precs,
#         best_result["se_precs"],
#         f1,
#         best_result["se_f1"],
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
#         best_result["se_accs"],
#         sens,
#         best_result["se_sens"],
#         specs,
#         best_result["se_specs"],
#         precs,
#         best_result["se_precs"],
#         f1,
#         best_result["se_f1"],
#     )
# )

# info.append(
#     "Alpha_h:{}, Alpha_p: {},neighbors:{}".format(best_ah, best_ap, args.n_neighbors)
# )
# info.append("Random state:{}".format(rand))
# filename = args.outputFile
# dirname = os.path.dirname(filename)
# if not os.path.exists(dirname):
#     os.makedirs(dirname)

# with open(filename, "w") as file_handler:
#     for item in info:
#         file_handler.write("{}\n".format(item))


#  waiting:
# python RASAR_df.py -i1 data/invivo/invivo_repeated_w_invitro.csv         -idf  data/invivo/datafusion_db_processed.csv -endpoint ['LC50','EC50'] -effect 'MOR'  -n 4 -ah logspace -ap logspace -dbi "overlap" -wi "True"  -il "both"        -o vivo+vitro(t&e)/bestR/repeat_both_df.txt
# python RASAR_df.py -i1 data/invivo/invivo_repeated_w_invitro.csv         -idf  data/invivo/datafusion_db_processed.csv -endpoint ['LC50','EC50'] -effect 'MOR'  -n 4 -ah logspace -ap logspace -dbi "overlap" -wi "own"   -il "both"        -o vivo+vitro(t&e)/bestR/repeat_own_both_df.txt
# python RASAR_df.py -i1 data/invivo/invivo_repeated_w_invitro_eawag.csv   -idf  data/invivo/datafusion_db_processed.csv -endpoint ['LC50','EC50'] -effect 'MOR'  -n 4 -ah logspace -ap logspace -dbi "overlap" -wi "True"  -il "both"        -o "vitro_e/vivo+vitro/bestR/repeat_both_df.txt"
# python RASAR_df.py -i1 data/invivo/invivo_repeated_w_invitro_eawag.csv   -idf  data/invivo/datafusion_db_processed.csv -endpoint ['LC50','EC50'] -effect 'MOR'  -n 4 -ah logspace -ap logspace -dbi "overlap" -wi "own"   -il "both"        -o "vitro_e/vivo+vitro/bestR/repeat_own_both_df.txt"
# python RASAR_df.py -i1 data/invivo/invivo_repeated_w_invitro_toxcast.csv -idf  data/invivo/datafusion_db_processed.csv -endpoint ['LC50','EC50'] -effect 'MOR'  -n 4 -ah logspace -ap logspace -dbi "overlap" -wi "True"  -il "both"        -o "vitro_t/vivo+vitro/bestR/repeat_both_df.txt"
# python RASAR_df.py -i1 data/invivo/invivo_repeated_w_invitro_toxcast.csv -idf  data/invivo/datafusion_db_processed.csv -endpoint ['LC50','EC50'] -effect 'MOR'  -n 4 -ah logspace -ap logspace -dbi "overlap" -wi "own"   -il "both"        -o "vitro_t/vivo+vitro/bestR/repeat_own_both_df.txt"
# python RASAR_df.py -i1 data/invivo/invivo_repeated_w_invitro.csv         -idf  data/invivo/datafusion_db_processed.csv -endpoint ['LC50','EC50'] -effect 'MOR'  -n 4 -ah logspace -ap logspace -dbi "overlap" -wi "True"  -il "both_half"   -o vivo+vitro(t&e)/bestR/repeat_both_df_half.txt
# python RASAR_df.py -i1 data/invivo/invivo_repeated_w_invitro.csv         -idf  data/invivo/datafusion_db_processed.csv -endpoint ['LC50','EC50'] -effect 'MOR'  -n 4 -ah logspace -ap logspace -dbi "overlap" -wi "own"   -il "both_half"   -o vivo+vitro(t&e)/bestR/repeat_own_both_df_half.txt
# python RASAR_df.py -i1 data/invivo/invivo_repeated_w_invitro_eawag.csv   -idf  data/invivo/datafusion_db_processed.csv -endpoint ['LC50','EC50'] -effect 'MOR'  -n 4 -ah logspace -ap logspace -dbi "overlap" -wi "True"  -il "both_half"   -o "vitro_e/vivo+vitro/bestR/repeat_both_df_half.txt"
# python RASAR_df.py -i1 data/invivo/invivo_repeated_w_invitro_eawag.csv   -idf  data/invivo/datafusion_db_processed.csv -endpoint ['LC50','EC50'] -effect 'MOR'  -n 4 -ah logspace -ap logspace -dbi "overlap" -wi "own"   -il "both_half"   -o "vitro_e/vivo+vitro/bestR/repeat_own_both_df_half.txt"
# python RASAR_df.py -i1 data/invivo/invivo_repeated_w_invitro_toxcast.csv -idf  data/invivo/datafusion_db_processed.csv -endpoint ['LC50','EC50'] -effect 'MOR'  -n 4 -ah logspace -ap logspace -dbi "overlap" -wi "True"  -il "both_half"   -o "vitro_t/vivo+vitro/bestR/repeat_both_df_half.txt"
# python RASAR_df.py -i1 data/invivo/invivo_repeated_w_invitro_toxcast.csv -idf  data/invivo/datafusion_db_processed.csv -endpoint ['LC50','EC50'] -effect 'MOR'  -n 4 -ah logspace -ap logspace -dbi "overlap" -wi "own"   -il "both_half"   -o "vitro_t/vivo+vitro/bestR/repeat_own_both_df_half.txt"


# new
