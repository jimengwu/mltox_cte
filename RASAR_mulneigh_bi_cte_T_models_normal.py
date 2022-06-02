import pandas as pd
from scipy.spatial.distance import jaccard, pdist, squareform, cdist
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import numpy as np
from time import ctime
from tqdm import tqdm
import argparse
from cte_helper_model import *
import multiprocessing as mp
from sklearn.model_selection import train_test_split, ParameterSampler
import pickle
import os
import sys
from pebble import ProcessPool
from concurrent.futures import TimeoutError
import traceback


import logging

# logging.debug("Debug message")
# logging.info("Info message")
# logging.warning("Warning message")
# logging.error("Error message")
# logging.critical("Critical message")


sys.modules["__main__"].__file__ = "ipython"
# db_invitro_matrix not defined in this code
db_invitro_matrix = None


def getArguments():
    parser = argparse.ArgumentParser(description="Running T_models for datasets.")
    parser.add_argument("-i", "--input", dest="inputFile", required=True)
    parser.add_argument(
        "-n", "--neighbors", dest="neighbors", required=True, nargs="+", type=int
    )
    parser.add_argument("-e", "--encoding", dest="encoding", default="binary")
    parser.add_argument("-ni", "--niter", dest="niter", default=5, type=int)
    parser.add_argument(
        "-il", "--invitro_label", dest="invitro_label", default="number"
    )
    parser.add_argument("-dbi", "--db_invitro", dest="db_invitro", default="noinvitro")
    parser.add_argument("-wi", "--w_invitro", dest="w_invitro", default="False")
    parser.add_argument("-ah", "--alpha_h", dest="hamming_alpha", default="logspace")
    parser.add_argument("-ap", "--alpha_p", dest="pubchem2d_alpha", default="logspace")
    parser.add_argument("-r", "--repeat", dest="repeat", default=20, type=int)
    parser.add_argument("-t_ls", "--t_ls", dest="t_ls", default="median")

    parser.add_argument("-o", "--output", dest="outputFile", default="binary.txt")
    return parser.parse_args()


args = getArguments()


def get_traintest_matrices(matrices, fold, ah, ap):
    matrix_euc = pd.DataFrame(matrices["euc"])
    max_euc = matrix_euc.iloc[fold[0], fold[0]].values.max()

    distance_matrix = pd.DataFrame(
        ah * matrices["hamming"]
        + ap * matrices["pubchem"]
        + matrix_euc.divide(max_euc).values
    )

    train_matrix = distance_matrix.iloc[fold[0], fold[0]]
    test_matrix = distance_matrix.iloc[fold[1], fold[0]]
    return train_matrix, test_matrix


def func(
    train_index,
    test_index,
    dist_matr_train,
    dist_matr_test,
    y_train,
    y_test,
    X,
    n_neighbors,
    ah,
    ap,
    num,
    model,
):

    if args.w_invitro == "own":
        train_rf = pd.DataFrame()
        test_rf = pd.DataFrame()
    else:
        train_rf, test_rf = cal_s_rasar(
            dist_matr_train,
            dist_matr_test,
            y_train,
            n_neighbors,
            args.encoding,
        )

    if args.w_invitro != "False":
        if str(args.db_invitro) == "overlap":
            train_rf = get_vitroinfo(train_rf, X, train_index, args.invitro_label)
            test_rf = get_vitroinfo(test_rf, X, test_index, args.invitro_label)
        else:
            train_rf = find_nearest_vitro(
                train_rf,
                args.db_invitro,
                db_invitro_matrix,
                train_index,
                args.invitro_label,
            )
            test_rf = find_nearest_vitro(
                test_rf,
                args.db_invitro,
                db_invitro_matrix,
                test_index,
                args.invitro_label,
            )

    df_score = fit_and_predict(
        model,
        train_rf,
        y_train,
        test_rf,
        y_test,
        encoding,
    )
    return df_score


def cal_matrixs(X1, X2, categorical, non_categorical):
    matrices = {}
    basic_mat = euclidean_matrix(X1, X2, non_categorical)
    matrix_h = hamming_matrix(X1, X2, categorical)
    matrix_p = pubchem2d_matrix(X1, X2)
    matrices["euc"] = basic_mat
    matrices["hamming"] = matrix_h
    matrices["pubchem"] = matrix_p
    return matrices


conc_column = "conc1_mean"

if args.encoding == "binary":
    encoding = "binary"
    encoding_value = 1
elif args.encoding == "multiclass":
    encoding = "multiclass"
    encoding_value = [0.1, 1, 10, 100]

# loading data & splitting into train and test dataset
print("loading dataset...", ctime())
db_mortality = load_data(
    args.inputFile,
    encoding=encoding,
    categorical_columns=categorical,
    conc_column=conc_column,
    encoding_value=encoding_value,
)
df_fishchem = db_mortality[["fish", "test_cas", conc_column]]
X = db_mortality.drop(columns=conc_column)
print("finish loaded.", ctime())

record = []
for repeat in range(args.repeat):
    grid_search = pd.DataFrame()
    logging.basicConfig(
        filename=args.outputFile + "_log_{}.log".format(repeat),
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
    )
    logging.info("The" + str(repeat) + "th times:")

    # train valid dataset splitting
    trainvalid_idx, valid_idx = get_grouped_train_test_split(
        df_fishchem,
        test_size=0.2,
        col_groups="test_cas",
        rand=repeat,
    )
    if list(valid_idx) in record:
        continue
    else:
        record.append(list(valid_idx))

    X_trainvalid, X_valid, Y_trainvalid, Y_valid = get_train_test_data(
        db_mortality, trainvalid_idx, valid_idx, conc_column
    )

    print("calcultaing distance matrix..", ctime())

    matrices_trainvalid = cal_matrixs(
        X_trainvalid, X_trainvalid, categorical, non_categorical
    )
    print("distance matrix calculation finished", ctime())

    # hyperparameter range
    if args.hamming_alpha == "logspace":
        sequence_ap = np.logspace(-2, 0, 20)
        sequence_ah = sequence_ap
    elif args.hamming_alpha == "best":
        result = pd.read_csv(
            "/local/wujimeng/code_jimeng/c_cte/V_models/best_alphas_"
            + str(repeat)
            + ".txt"
        )
        result = result.sort_values(
            by=[
                "train_avg_accuracy",
                "test_accuracy",
            ],
            ascending=False,
        )
        sequence_ah = [result.iloc[0].ah]
        sequence_ap = [result.iloc[0].ap]
        print(sequence_ah, sequence_ap)

    else:
        sequence_ap = [float(args.pubchem2d_alpha)]
        sequence_ah = [float(args.hamming_alpha)]

    if args.t_ls == "median":

        threshold_ls = X_trainvalid["invitro_conc"].quantile([0.5]).values
    else:

        threshold_ls = [
            pd.read_csv(
                "/local/wujimeng/code_jimeng/c_cte/GitHub/T_models/T_null/repeat_own_label_"
                + str(0)
                + ".txt"
            )
            .sort_values(
                by=["train_dataset_accuracy", "test_dataset_accuracy"],
                ascending=False,
            )
            .threshold.values[0]
        ]
    print(threshold_ls)
    hyper_params_tune = {
        # "n_estimators": [int(x) for x in np.linspace(start=1, stop=5, num=5)],
        # "max_features": [i for i in range(1, 4, 1)],
        # "max_depth": [i for i in np.linspace(start=0.01, stop=1, num=5)],
        # "min_samples_leaf": [i for i in np.linspace(start=0.01, stop=0.5, num=5)],
        # "min_samples_split": [i for i in np.linspace(start=0.1, stop=1, num=5)],
        # ---------------------------------
        # "n_estimators": [int(x) for x in np.linspace(start=1, stop=150, num=5)],
        # # "max_features": [i for i in range(1, 4, 1)],
        # "max_depth": [i for i in range(1, 10, 2)]
        # + [i for i in range(10, 100, 20)]
        # + [None],
        # "min_samples_leaf": [i for i in np.linspace(start=0.1, stop=0.5, num=5)],
        # "min_samples_split": [i for i in np.linspace(start=0.1, stop=1, num=5)],
        # "max_samples": [i for i in np.linspace(start=0.1, stop=1, num=5)],
        # --------------------------------------------
        "n_estimators": [int(x) for x in np.linspace(start=10, stop=100, num=5)]
        + [int(x) for x in np.linspace(start=2, stop=5, num=4)],
        "max_depth": [i for i in np.linspace(start=0.01, stop=1, num=5)]
        + [i for i in range(2, 10, 1)]
        + [i for i in range(10, 100, 20)]
        + [None],
        "min_samples_leaf": [i for i in np.linspace(start=0.01, stop=0.5, num=5)]
        + [i for i in np.linspace(start=0.1, stop=0.4, num=4)]
        + [1, 2, 4],
        "min_samples_split": [i for i in np.linspace(start=0.1, stop=1, num=5)]
        + [2, 4, 6],
        "max_samples": [i for i in np.linspace(start=0.1, stop=1, num=5)]
        + [i for i in np.linspace(start=0.2, stop=0.9, num=8)],
        # ---------------------------------------------
        # "n_estimators": [3],
        # "max_features": [1],
        # "max_depth": [1],
        # #     "criterion":["gini", "entropy"],
        # "min_samples_leaf": [0.255],
        # "min_samples_split": [0.1],
    }

    params_comb = list(
        ParameterSampler(
            hyper_params_tune,
            n_iter=args.niter,
            random_state=2,
        )
    )
    print(len(params_comb) * len(threshold_ls))
    # cross validation on the training dataset
    group_kfold = GroupKFold(n_splits=5)
    # group_kfold.get_n_splits()

    df_fishchem_tv = df_fishchem.iloc[trainvalid_idx, :]
    folds = list(group_kfold.split(df_fishchem_tv, groups=df_fishchem_tv["test_cas"]))

    count = 1
    import datetime

    starttime = datetime.datetime.now()
    num_runs = (
        len(sequence_ap) ** 2
        * len(params_comb)
        * len(args.neighbors)
        * len(threshold_ls)
    )
    for nei in args.neighbors:
        best_accs = 0
        grid_search = pd.DataFrame()
        for ah in sequence_ah:
            for ap in sequence_ap:
                for t in threshold_ls:

                    X_trainvalid = conc_to_label(X_trainvalid, t)
                    X_valid = conc_to_label(X_valid, t)
                    X = conc_to_label(X, t)
                    for j in range(0, len(params_comb)):
                        model = RandomForestClassifier(random_state=1)
                        for k, v in params_comb[j].items():
                            setattr(model, k, v)

                        try:
                            remain_time = (
                                (datetime.datetime.now() - starttime)
                                / count
                                * (num_runs - count)
                            )
                            print(
                                ah,
                                ",",
                                ap,
                                ",",
                                nei,
                                ",",
                                t,
                                "*" * 50,
                                count / num_runs,
                                remain_time,
                                ctime(),
                                end="\r",
                            )

                            count = count + 1
                            results = []
                            with ProcessPool(max_workers=5) as pool:
                                for num, fold in enumerate(folds):
                                    # print(len(fold[1]))
                                    y_train = Y_trainvalid[fold[0]]
                                    y_test = Y_trainvalid[fold[1]]

                                    if args.w_invitro == "own":
                                        train_matrix = pd.DataFrame()
                                        test_matrix = pd.DataFrame()
                                    else:
                                        (
                                            train_matrix,
                                            test_matrix,
                                        ) = get_traintest_matrices(
                                            matrices_trainvalid, fold, ah, ap
                                        )
                                    res = pool.schedule(
                                        func,
                                        args=(
                                            fold[0],
                                            fold[1],
                                            train_matrix,
                                            test_matrix,
                                            y_train,
                                            y_test,
                                            X_trainvalid,
                                            nei,
                                            ah,
                                            ap,
                                            num,
                                            model,
                                        ),
                                        timeout=120,
                                    )

                                    results.append(res)

                                    del train_matrix, test_matrix
                                results = [i.result() for i in results]

                            result_stats = result_describe(results)

                            train_index = X_trainvalid.index
                            test_index = X_valid.index
                            if args.w_invitro == "own":
                                train_rf = pd.DataFrame()
                                test_rf = pd.DataFrame()
                            else:

                                (
                                    matrix_trainvalid,
                                    matrix_test,
                                ) = get_traintest_matrices(
                                    matrices_full, [train_index, test_index], ah, ap
                                )

                                train_rf, test_rf = cal_s_rasar(
                                    matrix_trainvalid,
                                    matrix_test,
                                    Y_trainvalid,
                                    int(nei),
                                    encoding,
                                )

                            if args.w_invitro != "False":
                                if str(args.db_invitro) == "overlap":
                                    train_rf = get_vitroinfo(
                                        train_rf, X, train_index, args.invitro_label
                                    )
                                    test_rf = get_vitroinfo(
                                        test_rf, X, test_index, args.invitro_label
                                    )
                                else:

                                    db_invitro_matrix_new = normalized_invitro_matrix(
                                        X_trainvalid, db_invitro_matrix, ah, ap
                                    )

                                    train_rf = find_nearest_vitro(
                                        train_rf,
                                        args.db_invitro,
                                        db_invitro_matrix_new,
                                        train_index,
                                        args.invitro_label,
                                    )
                                    test_rf = find_nearest_vitro(
                                        test_rf,
                                        args.db_invitro,
                                        db_invitro_matrix_new,
                                        test_index,
                                        args.invitro_label,
                                    )

                            df_test_score = fit_and_predict(
                                model,
                                train_rf,
                                Y_trainvalid,
                                test_rf,
                                Y_valid,
                                args.encoding,
                            )
                            try:
                                dict_acc = dataset_acc(
                                    X_trainvalid, X_valid, Y_trainvalid, Y_valid
                                )
                            except:
                                dict_acc = pd.DataFrame()

                            temp_grid_full = save_results(
                                params_comb[j],
                                ah,
                                ap,
                                result_stats,
                                df_test_score,
                                dict_acc,
                            )
                            temp_grid_full["threshold"] = int(t)
                            grid_search = pd.concat([grid_search, temp_grid_full])
                            if result_stats[0].accuracy.values[0] > best_accs:
                                best_accs = result_stats[0].accuracy.values[0]
                                best_params = params_comb[j]
                                best_ah = ah
                                best_ap = ap
                                best_result = result_stats
                                print("success", best_accs)
                        except ValueError as error:
                            logging.error(str(error))
        print(best_result)
        df2file(grid_search, args.outputFile + "_{}_fullinfo.txt".format(repeat))
        del results

        # --------------testing---------------
        model = RandomForestClassifier(random_state=1)
        for k, v in best_params.items():
            setattr(model, k, v)
        train_index = X_trainvalid.index
        test_index = X_valid.index

        if args.w_invitro == "own":
            train_rf = pd.DataFrame()
            test_rf = pd.DataFrame()
        else:
            matrices_full = cal_matrixs(X, X, categorical, non_categorical)
            (matrix_trainvalid, matrix_test,) = get_traintest_matrices(
                matrices_full, [train_index, test_index], best_ah, best_ap
            )

            train_rf, test_rf = cal_s_rasar(
                matrix_trainvalid,
                matrix_test,
                Y_trainvalid,
                nei,
                encoding,
            )

        if args.w_invitro != "False":
            if str(args.db_invitro) == "overlap":
                train_rf = get_vitroinfo(train_rf, X, train_index, args.invitro_label)
                test_rf = get_vitroinfo(test_rf, X, test_index, args.invitro_label)
            else:

                db_invitro_matrix_new = normalized_invitro_matrix(
                    X_trainvalid, db_invitro_matrix, best_ah, best_ap
                )

                train_rf = find_nearest_vitro(
                    train_rf,
                    args.db_invitro,
                    db_invitro_matrix_new,
                    train_index,
                    args.invitro_label,
                )
                test_rf = find_nearest_vitro(
                    test_rf,
                    args.db_invitro,
                    db_invitro_matrix_new,
                    test_index,
                    args.invitro_label,
                )

        df_test_score = fit_and_predict(
            model,
            train_rf,
            Y_trainvalid,
            test_rf,
            Y_valid,
            args.encoding,
        )

        try:
            dict_acc = dataset_acc(X_trainvalid, X_valid, Y_trainvalid, Y_valid)
        except:
            pass

        temp_grid = save_results(
            best_params,
            best_ah,
            best_ap,
            best_result,
            df_test_score,
            dict_acc,
        )
        temp_grid["threshold"] = t
        # grid_search = pd.concat([grid_search, temp_grid])

    print("finished", ctime())
    # ----------------save the information into a file-------
    df2file(temp_grid, args.outputFile + "_{}.txt".format(repeat))


# ------------------------------------------------------get best alphas----------------------------------------------------------
# binary & R unrelated
# python RASAR_mulneigh_bi_cte_T_models.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -r 20 -n 1  -ni 25    -o "V_models/best_alphas"


# ------------------------------------------------------invitro + invivo to invivo(otv)----------------------------------------------------------
# binary & R unrelated

# python RASAR_mulneigh_bi_cte_T_models_normal.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -r 20 -ah 1  -ap 1 -n 1  -ni 5000000  -t_ls "median" -dbi "overlap" -wi "own"   -il "label"        -o "T_models/T_median_ml/repeat_own_label"
# python RASAR_mulneigh_bi_cte_T_models_normal.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -r 20 -ah 1  -ap 1 -n 1  -ni 5000000   -dbi "overlap" -wi "own"   -il "number"                     -o "T_models/T_num_ml/repeat_own_number"
# python RASAR_mulneigh_bi_cte_T_models_normal.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -r 20 -ah 1  -ap 1 -n 1  -ni 5000000   -t_ls "tuned"  -dbi "overlap" -wi "own"   -il "label"        -o "T_models/T_tuned_ml/repeat_own_label"


# cd /local/wujimeng/code_jimeng/c_cte/GitHub/
# source activate rdkit-test
