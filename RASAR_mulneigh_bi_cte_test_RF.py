import pandas as pd
from scipy.spatial.distance import jaccard, pdist, squareform, cdist
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from time import ctime
from tqdm import tqdm
import argparse
from helper_model_cte import *
import multiprocessing as mp
from sklearn.model_selection import train_test_split, ParameterSampler
import pickle
import os
import sys
from pebble import ProcessPool
from concurrent.futures import TimeoutError
import traceback

sys.modules["__main__"].__file__ = "ipython"
# db_invitro_matrix not defined in this code
db_invitro_matrix = None


def getArguments():
    parser = argparse.ArgumentParser(description="Running KNN_model for datasets.")
    parser.add_argument("-i", "--input", dest="inputFile", required=True)
    parser.add_argument(
        "-n", "--neighbors", dest="neighbors", required=True, nargs="+", type=int
    )
    parser.add_argument("-e", "--encoding", dest="encoding", default="binary")
    parser.add_argument("-iv", "--invitro", dest="invitro", default="False")
    parser.add_argument("-ni", "--niter", dest="niter", default=5, type=int)
    parser.add_argument(
        "-il", "--invitro_label", dest="invitro_label", default="number"
    )
    parser.add_argument("-dbi", "--db_invitro", dest="db_invitro", default="noinvitro")
    parser.add_argument("-wi", "--w_invitro", dest="w_invitro", default="False")
    parser.add_argument("-ah", "--alpha_h", dest="hamming_alpha", default="logspace")
    parser.add_argument("-ap", "--alpha_p", dest="pubchem2d_alpha", default="logspace")
    parser.add_argument("-r", "--repeat", dest="repeat", default=20, type=int)
    parser.add_argument("-o", "--output", dest="outputFile", default="binary.txt")
    return parser.parse_args()


args = getArguments()


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
    db_invitro_matrix,
    max_euc,
    num,
    model,
):

    # invitro = args.w_invitro
    # invitro_form = args.invitro_label
    # db_invitro = args.db_invitro

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
        # print("cal_s_rasar finished", end="\r")

    if args.w_invitro != "False":
        if str(args.db_invitro) == "overlap":
            train_rf = get_vitroinfo(train_rf, X, train_index, args.invitro_label)
            test_rf = get_vitroinfo(test_rf, X, test_index, args.invitro_label)
        else:
            db_invitro_matrix_new = pd.DataFrame(
                ah * db_invitro_matrix[0]
                + ap * db_invitro_matrix[1]
                + pd.DataFrame(db_invitro_matrix[2]).divide(max_euc).values
            )
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
    # print(train_rf.columns)
    df_score = fit_and_predict(
        model,
        train_rf,
        y_train,
        test_rf,
        y_test,
        encoding,
    )

    return df_score


def cal_normalized_matrix(X_trainvalid, X_valid, ah, ap):

    minmax = MinMaxScaler().fit(X_trainvalid[non_categorical])
    X_trainvalid[non_categorical] = minmax.transform(
        X_trainvalid.loc[:, non_categorical]
    )
    X_valid[non_categorical] = minmax.transform(X_valid.loc[:, non_categorical])

    matrix_valid = dist_matrix(
        X_valid,
        X_trainvalid,
        non_categorical,
        categorical,
        ah,
        ap,
    )
    matrix_trainvalid = dist_matrix(
        X_trainvalid,
        X_trainvalid,
        non_categorical,
        categorical,
        ah,
        ap,
    )
    return matrix_trainvalid, matrix_valid


def cal_normalized_matrix_invitro(X_trainvalid, db_invitro_matrix, ah, ap):
    max_euc = pd.DataFrame(
        euclidean_matrix(X_trainvalid, X_trainvalid, non_categorical)
    ).values.max()
    matrix_invitro_euc = pd.DataFrame(db_invitro_matrix[2])
    db_invitro_matrix_new = pd.DataFrame(
        ah * db_invitro_matrix[0]
        + ap * db_invitro_matrix[1]
        + matrix_invitro_euc.divide(max_euc).values
    )
    return db_invitro_matrix_new


def save_results(
    j,
    ah,
    ap,
    df_mean,
    df_std,
    df_test_score,
    X_trainvalid,
    X_valid,
    Y_trainvalid,
    Y_valid,
):
    temp_grid = pd.DataFrame()
    if j != None:
        for k, v in params_comb[j].items():
            temp_grid = pd.concat([temp_grid, pd.DataFrame([v], columns=[k])], axis=1)
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
    try:
        test_total = X_valid.shape[0]
        test_correct = np.sum(X_valid.invitro_label.values == Y_valid)
        test_acc = test_correct / test_total
        train_total = X_trainvalid.shape[0]
        train_correct = np.sum(X_trainvalid.invitro_label.values == Y_trainvalid)
        train_acc = train_correct / train_total
        temp_grid["train_dataset_accuracy"] = (
            str(round(train_acc, 4))
            + " ("
            + str(train_total)
            + "/"
            + str(train_correct)
            + ")"
        )

        temp_grid["test_dataset_accuracy"] = (
            str(round(test_acc, 4))
            + " ("
            + str(test_total)
            + "/"
            + str(test_correct)
            + ")"
        )

    except:
        pass

    return temp_grid


if __name__ == "__main__":
    conc_column = "conc1_mean"
    try:
        categorical, conc_column = get_col_ls(args.invitro)
    except:
        pass
    print(conc_column)
    if args.encoding == "binary":
        encoding = "binary"
        encoding_value = 1
    elif args.encoding == "multiclass":
        encoding = "multiclass"
        encoding_value = [0.1, 1, 10, 100]

    # rand = random.randrange(1, 100)
    rand = 2

    # loading data & splitting into train and test dataset
    print("loading dataset...", ctime())
    db_mortality = load_data(
        args.inputFile,
        encoding=encoding,
        categorical_columns=categorical,
        conc_column=conc_column,
        encoding_value=encoding_value,
        seed=rand,
    )
    # db_mortality = db_mortality[:200]
    print("finish loaded.", ctime())
    # db_mortality = db_mortality[:300]

    db_mortality["fish"] = (
        str(db_mortality["class"])
        + " "
        + str(db_mortality["tax_order"])
        + " "
        + str(db_mortality["family"])
        + " "
        + str(db_mortality["genus"])
        + " "
        + str(db_mortality["species"])
    )
    df_fishchem = db_mortality[["fish", "test_cas", conc_column]]
    # 10
    for repeat in range(args.repeat):
        filename = "vitro_e/vivo+vitro/R1/S/test/RF_tuned/log_" + str(repeat) + ".txt"

        grid_search = pd.DataFrame()

        # train valid dataset splitting

        test_size = 0.2
        col_groups = "test_cas"

        trainvalid_idx, valid_idx = get_grouped_train_test_split(
            df_fishchem, test_size, col_groups, rand=repeat
        )
        df_fishchem_tv = df_fishchem.iloc[trainvalid_idx, :]

        X = db_mortality.drop(columns=conc_column)

        X_trainvalid = X.iloc[trainvalid_idx, :]
        X_valid = X.iloc[valid_idx, :]
        Y_trainvalid = db_mortality.iloc[trainvalid_idx, :][conc_column].values
        Y_valid = db_mortality.iloc[valid_idx, :][conc_column].values

        print("calcultaing distance matrix..", ctime())
        matrix_euc, matrix_h, matrix_p = cal_matrixs(
            X_trainvalid, X_trainvalid, categorical, non_categorical
        )
        print("distance matrix calculation finished", ctime())

        # hyperparameter range
        if args.hamming_alpha == "logspace":
            sequence_ap = np.logspace(-2, 0, 20)
            # sequence_ap = np.logspace(-5, 0, 30)
            sequence_ah = sequence_ap
        else:
            sequence_ap = [float(args.pubchem2d_alpha)]
            sequence_ah = [float(args.hamming_alpha)]

        hyper_params_tune = {
            # "n_estimators": [int(x) for x in np.linspace(start=50, stop=800, num=10)],
            # "n_estimators": [int(x) for x in np.linspace(start=1, stop=100, num=5)],
            # "min_weight_fraction_leaf": [0.1, 0.2, 0.5],
            # "min_samples_split": [2,4, 6,8],
            # "min_samples_leaf": [1, 2, 4],
            # "max_samples": [0.1 * i for i in range(1, 10, 1)],
            # "max_depth": [i for i in range(1, 10, 1)],
            # "bootstrap": [True, False],
            "n_estimators": [int(x) for x in np.linspace(start=500, stop=2000, num=10)],
            "max_features": [i for i in range(1, 5, 1)],
            "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            "criterion": ["gini", "entropy"],
            "min_samples_leaf": [1, 2, 4],
            "min_samples_split": [2, 5, 10],
            # "max_samples": []
            # "class_weight": [{0: i, 1: 1} for i in range(0, 10)]
            # "class_weight": ["balanced"]
            # + [{0: i, 1: 1} for i in np.linspace(0, 10, 5)]
            # + [{0: i, 1: 1} for i in np.linspace(0, 1, 5)],
            # "class_weight": [{0: i, 1: 1} for i in range(0, 20)],
        }

        params_comb = list(
            ParameterSampler(
                hyper_params_tune,
                n_iter=args.niter,
                random_state=rand,
            )
        )

        # cross validation on the training dataset
        group_kfold = GroupKFold(n_splits=4)
        group_kfold.get_n_splits()
        folds = list(
            group_kfold.split(df_fishchem_tv, groups=df_fishchem_tv[col_groups])
        )

        count = 1

        for n in args.neighbors:
            avg_accs = 0
            for ah in sequence_ah:
                for ap in sequence_ap:
                    model = RandomForestClassifier(random_state=1)
                    try:
                        results = []

                        print(
                            ah,
                            ap,
                            "*" * 50,
                            count / (len(sequence_ap) ** 2),
                            ctime(),
                            end="\r",
                        )
                        count = count + 1

                        with ProcessPool(max_workers=4) as pool:
                            for num, fold in enumerate(folds):
                                y_train = Y_trainvalid[fold[0]]
                                y_test = Y_trainvalid[fold[1]]

                                matrix_euc = pd.DataFrame(matrix_euc)
                                max_euc = matrix_euc.iloc[fold[0], fold[0]].values.max()

                                distance_matrix = pd.DataFrame(
                                    ah * matrix_h
                                    + ap * matrix_p
                                    + matrix_euc.divide(max_euc).values
                                )

                                train_matrix = distance_matrix.iloc[fold[0], fold[0]]
                                test_matrix = distance_matrix.iloc[fold[1], fold[0]]

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
                                        n,
                                        ah,
                                        ap,
                                        db_invitro_matrix,
                                        max_euc,
                                        num,
                                        model,
                                    ),
                                    timeout=120,
                                )
                                try:
                                    results.append(res)
                                except TimeoutError as error:
                                    print(
                                        "function took longer than %d seconds"
                                        % error.args[1]
                                    )
                                del train_matrix, test_matrix
                            results = [i.result() for i in results]

                        # print("start testing", end="\r")
                        df_output = pd.concat(results, axis=0)
                        df_mean = pd.DataFrame(df_output.mean(axis=0)).transpose()
                        df_std = pd.DataFrame(df_output.sem(axis=0)).transpose()

                        train_index = X_trainvalid.index
                        test_index = X_valid.index

                        if args.w_invitro == "own":
                            train_rf = pd.DataFrame()
                            test_rf = pd.DataFrame()
                        else:
                            matrix_train, matrix_test = cal_normalized_matrix(
                                X_trainvalid, X_valid, ah, ap
                            )
                            train_rf, test_rf = cal_s_rasar(
                                matrix_train,
                                matrix_test,
                                Y_trainvalid,
                                n,
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
                                db_invitro_matrix_new = cal_normalized_matrix_invitro(
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

                        # print(train_rf.columns, ctime())

                        df_test_score = fit_and_predict(
                            model,
                            train_rf,
                            Y_trainvalid,
                            test_rf,
                            Y_valid,
                            args.encoding,
                        )

                        temp_grid = save_results(
                            None,
                            ah,
                            ap,
                            df_mean,
                            df_std,
                            df_test_score,
                            X_trainvalid,
                            X_valid,
                            Y_trainvalid,
                            Y_valid,
                        )

                        grid_search = pd.concat([grid_search, temp_grid])
                    except ValueError as error:
                        f = open(filename, "a")
                        f.write(str(error))
                        f.write("\n")
                        f.close()

                        continue

            df2file(
                grid_search, args.outputFile + "_{}" + "_bestalphas.txt".format(repeat)
            )
            best_ah = grid_search.sort_values(by=["train_avg_accuracy"]).ah.values[0]
            best_ap = grid_search.sort_values(by=["train_avg_accuracy"]).ap.values[0]
            for j in tqdm(range(0, len(params_comb))):
                model = RandomForestClassifier(random_state=1)
                for k, v in params_comb[j].items():
                    setattr(model, k, v)
                try:
                    results = []
                    count = count + 1
                    with ProcessPool(max_workers=4) as pool:
                        for num, fold in enumerate(folds):
                            y_train = Y_trainvalid[fold[0]]
                            y_test = Y_trainvalid[fold[1]]

                            matrix_euc = pd.DataFrame(matrix_euc)
                            max_euc = matrix_euc.iloc[fold[0], fold[0]].values.max()

                            distance_matrix = pd.DataFrame(
                                best_ah * matrix_h
                                + best_ap * matrix_p
                                + matrix_euc.divide(max_euc).values
                            )

                            train_matrix = distance_matrix.iloc[fold[0], fold[0]]
                            test_matrix = distance_matrix.iloc[fold[1], fold[0]]

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
                                    n,
                                    best_ah,
                                    best_ap,
                                    db_invitro_matrix,
                                    max_euc,
                                    num,
                                    model,
                                ),
                                timeout=120,
                            )
                            try:
                                results.append(res)
                            except TimeoutError as error:
                                print(
                                    "function took longer than %d seconds"
                                    % error.args[1]
                                )
                            del train_matrix, test_matrix
                        results = [i.result() for i in results]

                    # print("start testing", end="\r")
                    df_output = pd.concat(results, axis=0)
                    df_mean = pd.DataFrame(df_output.mean(axis=0)).transpose()
                    df_std = pd.DataFrame(df_output.sem(axis=0)).transpose()

                    train_index = X_trainvalid.index
                    test_index = X_valid.index

                    if args.w_invitro == "own":
                        train_rf = pd.DataFrame()
                        test_rf = pd.DataFrame()
                    else:
                        matrix_train, matrix_test = cal_normalized_matrix(
                            X_trainvalid, X_valid, best_ah, best_ap
                        )
                        train_rf, test_rf = cal_s_rasar(
                            matrix_train,
                            matrix_test,
                            Y_trainvalid,
                            n,
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
                            db_invitro_matrix_new = cal_normalized_matrix_invitro(
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

                    # print(train_rf.columns, ctime())

                    df_test_score = fit_and_predict(
                        model,
                        train_rf,
                        Y_trainvalid,
                        test_rf,
                        Y_valid,
                        args.encoding,
                    )

                    temp_grid = save_results(
                        j,
                        best_ah,
                        best_ap,
                        df_mean,
                        df_std,
                        df_test_score,
                        X_trainvalid,
                        X_valid,
                        Y_trainvalid,
                        Y_valid,
                    )

                    grid_search = pd.concat([grid_search, temp_grid])

                except ValueError as error:
                    # print(error)
                    f = open(filename, "a")
                    f.write(str(error))
                    f.write("\n")
                    f.close()

                    continue

        # ----------------save the information into a file-------
        df2file(grid_search, args.outputFile + "_{}.txt".format(repeat))


# -------------------------------------------------invivo to invivo (gvv)-------------------------------------------------
# binary:
# python RASAR_mulneigh_bi_cte_test_RF.py -i '/local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv' -r 1 -n 1 -ni 5000  -o "general/vivo/s_rasar_rasar_bi_1_v2"


# ------------------------------------------------------invitro to invitro(gtt)----------------------------------------------------------
# binary:
# python RASAR_mulneigh_bi_cte_test_RF.py     -i /local/wujimeng/code_jimeng/data/invitro/invitro_processed.csv  -r 1  -ni 5000  -iv eawag   -n 1  -o general/vitro/vitro_s_rasar_bi_1_v2


# cd /local/wujimeng/code_jimeng/c_cte
# source activate rdkit-test


# ------------------------------------------------------invitro + invivo to invivo(otv)----------------------------------------------------------
# binary & R unrelated
# python RASAR_mulneigh_bi_cte_test_RF.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv  -ah 1  -ap 1 -n 1  -ni 20000   -dbi "overlap" -wi "own"   -il "label"                -o "vitro_e/vivo+vitro/R1/S/test/RF_tuned/try2/repeat_own_label"
# python RASAR_mulneigh_bi_cte_test_RF.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv  -ah 1  -ap 1 -n 1  -ni 20000   -dbi "overlap" -wi "own"   -il "number"               -o "vitro_e/vivo+vitro/R1/S/test/RF_tuned/try2/repeat_own_number"
# python RASAR_mulneigh_bi_cte_test_RF.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv  -ah 1  -ap 1 -n 1  -ni 20000   -dbi "overlap" -wi "own"   -il "label_half"           -o "vitro_e/vivo+vitro/R1/S/test/RF_tuned/try2/repeat_own_label_half"


# ------------------------------------------------------invitro + invivo to invivo(otvv)----------------------------------------------------------
# binary & R=1
# python RASAR_mulneigh_bi_cte_test_LR.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv  -n 1  -ni 20000 -dbi "overlap" -wi "True"   -il "label"                -o "vitro_e/vivo+vitro/R1/S/test/LR/repeat_label"

# python RASAR_mulneigh_bi_cte_test_RF.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv  -n 1  -ni 20000 -ah 0.143845 -ap 0.069519  -dbi "overlap" -wi "True"   -il "label"                -o "vitro_e/vivo+vitro/R1/S/test/RF_tuned/try2/repeat_label"
# python RASAR_mulneigh_bi_cte_test_RF.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv  -n 1  -ni 20000 -ah 0.143845 -ap 0.069519 -dbi "overlap" -wi "True"   -il "number"               -o "vitro_e/vivo+vitro/R1/S/test/RF_tuned/try2/repeat_number"
# python RASAR_mulneigh_bi_cte_test_RF.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv  -n 1  -ni 20000 -ah 0.143845 -ap 0.069519 -dbi "overlap" -wi "True"   -il "label_half"           -o "vitro_e/vivo+vitro/R1/S/test/RF_tuned/try2/repeat_label_half"

# binary & R=4
# python RASAR_mulneigh_bi_cte_test_RF.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv  -n 4     -dbi "overlap" -wi "True"   -il "label"                -o "vitro_e/vivo+vitro/R4/S/test/RF/repeat_label"
# python RASAR_mulneigh_bi_cte_test_RF.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv  -n 4     -dbi "overlap" -wi "True"   -il "number"               -o "vitro_e/vivo+vitro/R4/S/test/RF/repeat_number"
# python RASAR_mulneigh_bi_cte_test_RF.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv  -n 4     -dbi "overlap" -wi "True"   -il "label_half"           -o "vitro_e/vivo+vitro/R4/S/test/RF/repeat_label_half"


# ------------------------------------------------------invivo to sinvivo(ovv)----------------------------------------------------------
# binary & R=1
# python RASAR_mulneigh_bi_cte_test_RF.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -ah 0.143845 -ap 0.069519 -n 1 -ni 30000  -o "vitro_e/vivo+vitro/R1/S/test/RF_tuned/try2/repeat"

# binary & R=4
# python RASAR_mulneigh_bi_cte_test_RF.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv  -n 4 -ni 20000  -o "vitro_e/vivo+vitro/R4/S/test/RF/repeat"
