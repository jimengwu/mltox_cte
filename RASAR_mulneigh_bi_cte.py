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
from cte_helper_model import *
import multiprocessing as mp
from sklearn.model_selection import train_test_split, ParameterSampler
import pickle
import os
import datetime

# db_invitro_matrix not defined in this code
db_invitro_matrix = None


def getArguments():
    parser = argparse.ArgumentParser(description="Running KNN_model for datasets.")
    parser.add_argument("-i", "--input", dest="inputFile", required=True)
    parser.add_argument("-n", "--neighbors", dest="neighbors", required=True, nargs="+")
    parser.add_argument("-e", "--encoding", dest="encoding", default="binary")
    parser.add_argument("-iv", "--invitro", dest="invitro", default="False")
    parser.add_argument("-ni", "--niter", dest="niter", default=50, type=int)
    parser.add_argument(
        "-il", "--invitro_label", dest="invitro_label", default="number"
    )
    parser.add_argument("-dbi", "--db_invitro", dest="db_invitro", default="noinvitro")
    parser.add_argument("-wi", "--w_invitro", dest="w_invitro", default="False")
    parser.add_argument("-ah", "--alpha_h", dest="hamming_alpha", default="logspace")
    parser.add_argument("-ap", "--alpha_p", dest="pubchem2d_alpha", default="logspace")

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


if __name__ == "__main__":
    conc_column = "conc1_mean"

    if args.invitro == "both":
        categorical = ["class", "tax_order", "family", "genus", "species"]
    elif args.invitro == "eawag":
        categorical = [
            "class",
            "tax_order",
            "family",
            "genus",
            "species",
            "cell_line",
            "endpoint",
            "solvent",
            "conc_determination_nominal_or_measured",
        ]
        conc_column = "ec50"
    elif args.invitro == "toxcast":
        categorical = [
            "class",
            "tax_order",
            "family",
            "genus",
            "species",
            # "modl"
        ]
        conc_column = "conc"

    if args.encoding == "binary":
        encoding = "binary"
        encoding_value = 1
    elif args.encoding == "multiclass":
        encoding = "multiclass"
        encoding_value = [0.1, 1, 10, 100]

    rand = random.randrange(1, 100)
    rand = 2

    # ----------loading data & splitting into train and test dataset----
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
    X = db_mortality.drop(columns=conc_column)
    print("finish loaded.", ctime())

    # -------train valid dataset splitting------
    test_size = 0.2
    col_groups = "test_cas"

    df_fishchem = db_mortality[["fish", "test_cas", conc_column]]
    trainvalid_idx, valid_idx = get_grouped_train_test_split(
        df_fishchem, test_size, col_groups
    )
    df_fishchem_tv = df_fishchem.iloc[trainvalid_idx, :]

    X_trainvalid, X_valid, Y_trainvalid, Y_valid = get_train_test_data(
        db_mortality, trainvalid_idx, valid_idx, conc_column
    )
    #
    if args.w_invitro != "own":
        print("calcultaing distance matrix..", ctime())

        matrices_trainvalid = cal_matrixs(
            X_trainvalid, X_trainvalid, categorical, non_categorical
        )
        matrices_full = cal_matrixs(X, X, categorical, non_categorical)
        print("distance matrix calculation finished", ctime())

    # --------hyperparameter range--------
    if args.hamming_alpha == "logspace":
        sequence_ap = np.logspace(-2, 0, 20)
        # sequence_ap = np.logspace(-5, 0, 30)
        sequence_ah = sequence_ap
    else:
        sequence_ap = [float(args.pubchem2d_alpha)]
        sequence_ah = [float(args.hamming_alpha)]

    hyper_params_tune = {
        "n_estimators": [int(x) for x in np.linspace(start=200, stop=500, num=6)],
        "max_depth": [i for i in range(10, 36, 5)],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 4, 8, 16],
        # "class_weight": [{0: i, 1: 1} for i in range(0, 10)]
        "class_weight": ["balanced"]
        + [{0: i, 1: 1} for i in np.linspace(0, 10, 5)]
        + [{0: i, 1: 1} for i in np.linspace(0, 1, 5)],
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
    folds = list(group_kfold.split(df_fishchem_tv, groups=df_fishchem_tv[col_groups]))

    count = 1
    print(len(params_comb))

    starttime = datetime.datetime.now()
    num_runs = len(sequence_ap) ** 2 * len(params_comb) * len(args.neighbors)
    for nei in args.neighbors:
        best_accs = 0
        grid_search = pd.DataFrame()
        for ah in sequence_ah:
            for ap in sequence_ap:
                for j in range(0, len(params_comb)):
                    model = RandomForestClassifier(random_state=1)
                    for k, v in params_comb[j].items():
                        setattr(model, k, v)
                    results = []

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
                        "*" * 50,
                        count / num_runs,
                        remain_time,
                        "remained.",
                        ctime(),
                        end="\r",
                    )
                    count = count + 1
                    with mp.Pool(4) as pool:
                        for num, fold in enumerate(folds):
                            y_train = Y_trainvalid[fold[0]]
                            y_test = Y_trainvalid[fold[1]]
                            if args.w_invitro == "own":
                                train_matrix = pd.DataFrame()
                                test_matrix = pd.DataFrame()
                            else:
                                (train_matrix, test_matrix,) = get_traintest_matrices(
                                    matrices_trainvalid, fold, ah, ap
                                )

                            res = pool.apply_async(
                                func,
                                args=(
                                    fold[0],
                                    fold[1],
                                    train_matrix,
                                    test_matrix,
                                    y_train,
                                    y_test,
                                    X_trainvalid,
                                    int(nei),
                                    model,
                                ),
                            )
                            results.append(res)
                            del res, train_matrix, test_matrix
                        results = [res.get() for res in results]

                    result_stats = result_describe(results)

                    train_index = X_trainvalid.index
                    test_index = X_valid.index
                    if args.w_invitro == "own":
                        train_rf = pd.DataFrame()
                        test_rf = pd.DataFrame()
                    else:

                        (matrix_trainvalid, matrix_test,) = get_traintest_matrices(
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
                        model, train_rf, Y_trainvalid, test_rf, Y_valid, args.encoding
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
                    temp_grid_full["num_neighbor"] = int(nei)
                    grid_search = pd.concat([grid_search, temp_grid_full])
                    # get the best hyperparameters
                    if (
                        np.mean([results[k]["accuracy"] for k in range(len(results))])
                        > best_accs
                    ):

                        best_result = result_stats
                        best_params = params_comb[j]
                        best_accs = result_stats[0].accuracy.values[0]
                        best_ah = ah
                        best_ap = ap
                        print(best_accs, ah, ap, "success!")
        print(best_result)
        df2file(grid_search, args.outputFile + "_{}_fullinfo.txt".format(nei))
        del results

        # --------------test on the test dataset----------------
        model = RandomForestClassifier(random_state=1)
        for k, v in best_params.items():
            setattr(model, k, v)

        train_index = X_trainvalid.index
        test_index = X_valid.index

        if args.w_invitro == "own":
            train_rf = pd.DataFrame()
            test_rf = pd.DataFrame()
        else:

            (matrix_trainvalid, matrix_test,) = get_traintest_matrices(
                matrices_full, [train_index, test_index], best_ah, best_ap
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

        print(train_rf.columns, ctime())

        df_test_score = fit_and_predict(
            model, train_rf, Y_trainvalid, test_rf, Y_valid, args.encoding
        )
        try:
            dict_acc = dataset_acc(X_trainvalid, X_valid, Y_trainvalid, Y_valid)
        except:
            dict_acc = pd.DataFrame()

        temp_grid = save_results(
            best_params,
            best_ah,
            best_ap,
            best_result,
            df_test_score,
            dict_acc,
        )
        temp_grid["num_neighbor"] = int(nei)
        # ----------------save the information into a file-------
        print("finished", ctime())
        df2file(temp_grid, args.outputFile + "_{}.txt".format(nei))


# python RASAR_mulneigh_bi_cte.py -i '/local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv' -n 1 2 3   -o n_mul/s_rasar_rasar_bi

# -------------------------------------------------invivo to invivo (gvv)-------------------------------------------------
# binary:
# python RASAR_mulneigh_bi_cte.py -i '/local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv' -n 1   -o n_mul/balanced/s_rasar_rasar_bi_1.csv
# python RASAR_mulneigh_bi_cte.py -i '/local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv' -n 5   -o n_mul/balanced/s_rasar_rasar_bi_5.csv
# python RASAR_mulneigh_bi_cte.py -i '/local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv' -n 4   -o n_mul/balanced/s_rasar_rasar_bi_4.csv
# python RASAR_mulneigh_bi_cte.py -i '/local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv' -n 2   -o n_mul/balanced/s_rasar_rasar_bi_2.csv
# python RASAR_mulneigh_bi_cte.py -i '/local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv' -n 3   -o n_mul/balanced/s_rasar_rasar_bi_3.csv

# python RASAR_mulneigh_bi_cte.py -i '/local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv' -n 7   -o n_mul/balanced/s_rasar_rasar_bi_7.csv
# python RASAR_mulneigh_bi_cte.py -i '/local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv' -n 9   -o n_mul/balanced/s_rasar_rasar_bi_9.csv
# python RASAR_mulneigh_bi_cte.py -i '/local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv' -n 11   -o n_mul/balanced/s_rasar_rasar_bi_11.csv
# python RASAR_mulneigh_bi_cte.py -i '/local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv' -n 13   -o n_mul/balanced/s_rasar_rasar_bi_13.csv
# python RASAR_mulneigh_bi_cte.py -i '/local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv' -n 15   -o n_mul/balanced/s_rasar_rasar_bi_15.csv

# multiclass:
# python RASAR_mulneigh_bi.py -i 'data/invivo/lc50_processed_jim.csv' -n 1  -e "multiclass" -o bestR/s_rasar_rasar_mul_1.csv
# python RASAR_mulneigh_bi.py -i 'data/invivo/lc50_processed_jim.csv' -n 5  -e "multiclass" -o bestR/s_rasar_rasar_mul_5.csv
# python RASAR_mulneigh_bi.py -i 'data/invivo/lc50_processed_jim.csv' -n 4  -e "multiclass" -o bestR/s_rasar_rasar_mul_4.csv
# python RASAR_mulneigh_bi.py -i 'data/invivo/lc50_processed_jim.csv' -n 2  -e "multiclass" -o bestR/s_rasar_rasar_mul_2.csv
# python RASAR_mulneigh_bi.py -i 'data/invivo/lc50_processed_jim.csv' -n 3  -e "multiclass" -o bestR/s_rasar_rasar_mul_3.csv
# python RASAR_mulneigh_bi.py -i 'data/invivo/lc50_processed_jim.csv' -n 7  -e "multiclass" -o bestR/s_rasar_rasar_mul_7.csv
# python RASAR_mulneigh_bi.py -i 'data/invivo/lc50_processed_jim.csv' -n 9  -e "multiclass" -o bestR/s_rasar_rasar_mul_9.csv
# python RASAR_mulneigh_bi.py -i 'data/invivo/lc50_processed_jim.csv' -n 11  -e "multiclass" -o bestR/s_rasar_rasar_mul_11.csv
# python RASAR_mulneigh_bi.py -i 'data/invivo/lc50_processed_jim.csv' -n 13  -e "multiclass" -o bestR/s_rasar_rasar_mul_13.csv
# python RASAR_mulneigh_bi.py -i 'data/invivo/lc50_processed_jim.csv' -n 15  -e "multiclass" -o bestR/s_rasar_rasar_mul_15.csv


# ------------------------------------------------------invitro to invitro(gtt)----------------------------------------------------------
# binary:
# python RASAR_mulneigh_bi_cte.py     -i /local/wujimeng/code_jimeng/data/invitro/invitro_processed.csv      -iv eawag   -n 4  -o "vitro_e/vitro/vitro_s_rasar_bi_4.txt"
# python RASAR_mulneigh_bi_cte.py     -i /local/wujimeng/code_jimeng/data/invitro/invitro_processed.csv      -iv eawag   -n 1  -o vitro_e/vitro/vitro_s_rasar_bi_1.txt

# multiclass & R=5:
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invitro/invitro_processed.csv      -e "multiclass"    -iv eawag   -n 5  -o "vitro_e/vitro/vitro_s_rasar_mul_5.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invitro/invitro_processed.csv      -e "multiclass"    -iv eawag   -n 1  -o vitro_e/vitro/vitro_s_rasar_mul_1.txt


# ------------------------------------------------------invitro to invitro(ott)----------------------------------------------------------
# binary:
# python RASAR_mulneigh_bi_cte.py -i /local/wujimeng/code_jimeng/data/invitro/invitro_eawag_repeated.csv -iv eawag   -n 4  -o "vitro_e/vitro/vitro_s_rasar_bi_repeated_4.txt"
# python RASAR_mulneigh_bi_cte.py -i /local/wujimeng/code_jimeng/data/invitro/invitro_eawag_repeated.csv -iv eawag   -n 1  -o "vitro_e/vitro/vitro_s_rasar_bi_repeated_1.txt"

# multiclass & R=5:
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invitro/invitro_eawag_repeated.csv -e "multiclass"    -iv eawag   -n 5  -o "vitro_e/vitro/vitro_s_rasar_mul_repeated_5.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invitro/invitro_eawag_repeated.csv -e "multiclass"    -iv eawag   -n 1  -o vitro_e/vitro/vitro_s_rasar_mul_repeated_1.txt


# ------------------------------------------------------invivo to invivo(ovv)----------------------------------------------------------
# binary & R=4
# python RASAR_mulneigh_bi_cte.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 4 -dbi "no"   -wi "False" -il "both"      -o  "vitro_e/vivo/vivo_s_rasar_bi_repeated_4.txt"

# multiclass & & R=4, not enough data, cannot use R = 5
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 4 -e "multiclass" -ah "logspace" -ap "logspace" -dbi "no"      -wi "False" -il "both"    -o "vitro_e/vivo+vitro/bestR/multiclass/S/repeat_mul.txt"


# ------------------------------------------------------invitro + invivo to invivo(otv)----------------------------------------------------------
# binary & R=4
# python RASAR_mulneigh_bi_cte.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv  -ah 1  -ap 1   -ni 20000   -dbi "overlap" -wi "own"   -il "number"               -o "vitro_e/vivo+vitro/bestR/S/repeat_own_number.txt"
# python RASAR_mulneigh_bi_cte.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv  -ah 1  -ap 1   -ni 20000   -dbi "overlap" -wi "own"   -il "label"                -o "vitro_e/vivo+vitro/bestR/S/repeat_own_label.txt"
# python RASAR_mulneigh_bi_cte.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv  -ah 1  -ap 1   -ni 20000   -dbi "overlap" -wi "own"   -il "label_half"           -o "vitro_e/vivo+vitro/bestR/S/repeat_own_label_half.txt"


# ------------------------------------------------------invitro + invivo to invivo(otvv)----------------------------------------------------------
# binary & R=4

# python RASAR_mulneigh_bi_cte.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "number"               -o "vitro_e/vivo+vitro/bestR/S/repeat_number.txt"
# python RASAR_mulneigh_bi_cte.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "label"                -o "vitro_e/vivo+vitro/bestR/S/repeat_label.txt"
# python RASAR_mulneigh_bi_cte.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "label_half"           -o "vitro_e/vivo+vitro/bestR/S/repeat_label_half.txt"
# python RASAR_mulneigh_bi_cte.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 1 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "number"               -o "vitro_e/vivo+vitro/R1/S/repeat_number.txt"
# python RASAR_mulneigh_bi_cte.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 1 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "label"                -o "vitro_e/vivo+vitro/R1/S/repeat_label.txt"
# python RASAR_mulneigh_bi_cte.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 1 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "label_half"           -o "vitro_e/vivo+vitro/R1/S/repeat_label_half.txt"


# multiclass & R=4, not enough data, cannot use R = 5
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 4 -e "multiclass" -ah "logspace" -ap "logspace" -dbi "overlap" -wi "own"   -il "number"  -o "vitro_e/vivo+vitro/bestR/multiclass/S/repeat_own_number_mul.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 4 -e "multiclass" -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "number"  -o "vitro_e/vivo+vitro/bestR/multiclass/S/repeat_number_mul.txt"


# binary & R=4, all other results:
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "both_half"            -o "vitro_e/vivo+vitro/bestR/S/repeat_both_half.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 4 -ah 1          -ap 1          -dbi "overlap" -wi "own"   -il "both"                 -o "vitro_e/vivo+vitro/bestR/S/repeat_own_both.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "both"                 -o "vitro_e/vivo+vitro/bestR/S/repeat_both.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 4 -ah 1          -ap 1          -dbi "overlap" -wi "own"   -il "both_half"            -o "vitro_e/vivo+vitro/bestR/S/repeat_own_both_half.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 4 -ah 1          -ap 1          -dbi "overlap" -wi "own"   -il "both_reserved"        -o "vitro_e/vivo+vitro/bestR/S/repeat_own_both_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 4 -ah 1          -ap 1          -dbi "overlap" -wi "own"   -il "label_reserved"       -o "vitro_e/vivo+vitro/bestR/S/repeat_own_label_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "label_reserved"       -o "vitro_e/vivo+vitro/bestR/S/repeat_label_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "label_half_reserved"  -o "vitro_e/vivo+vitro/bestR/S/repeat_label_half_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 4 -ah 1          -ap 1          -dbi "overlap" -wi "own"   -il "label_half_reserved"  -o "vitro_e/vivo+vitro/bestR/S/repeat_own_label_half_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 4 -ah 1          -ap 1          -dbi "overlap" -wi "own"   -il "both_half_reserved"   -o "vitro_e/vivo+vitro/bestR/S/repeat_own_both_half_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "both_half_reserved"   -o "vitro_e/vivo+vitro/bestR/S/repeat_both_half_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "both_reserved"        -o "vitro_e/vivo+vitro/bestR/S/repeat_both_reserved.txt"

# binary & R=1:
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 1 -ah "logspace" -ap "logspace" -dbi "no"      -wi "False" -il "both"                 -o "vitro_e/vivo+vitro/R1/S/repeat.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 1 -ah 1          -ap 1          -dbi "overlap" -wi "own"   -il "number"               -o "vitro_e/vivo+vitro/R1/S/repeat_own_number.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 1 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "number"               -o "vitro_e/vivo+vitro/R1/S/repeat_number.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 1 -ah 1          -ap 1          -dbi "overlap" -wi "own"   -il "label"                -o "vitro_e/vivo+vitro/R1/S/repeat_own_label.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 1 -ah 1          -ap 1          -dbi "overlap" -wi "own"   -il "label_half"           -o "vitro_e/vivo+vitro/R1/S/repeat_own_label_half.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 1 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "label"                -o "vitro_e/vivo+vitro/R1/S/repeat_label.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 1 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "label_half"           -o "vitro_e/vivo+vitro/R1/S/repeat_label_half.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 1 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "both_half"            -o "vitro_e/vivo+vitro/R1/S/repeat_both_half.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 1 -ah 1          -ap 1          -dbi "overlap" -wi "own"   -il "both"                 -o "vitro_e/vivo+vitro/R1/S/repeat_own_both.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 1 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "both"                 -o "vitro_e/vivo+vitro/R1/S/repeat_both.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 1 -ah 1          -ap 1          -dbi "overlap" -wi "own"   -il "both_half"            -o "vitro_e/vivo+vitro/R1/S/repeat_own_both_half.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 1 -ah 1          -ap 1          -dbi "overlap" -wi "own"   -il "both_reserved"        -o "vitro_e/vivo+vitro/R1/S/repeat_own_both_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 1 -ah 1          -ap 1          -dbi "overlap" -wi "own"   -il "label_reserved"       -o "vitro_e/vivo+vitro/R1/S/repeat_own_label_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 1 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "label_reserved"       -o "vitro_e/vivo+vitro/R1/S/repeat_label_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 1 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "label_half_reserved"  -o "vitro_e/vivo+vitro/R1/S/repeat_label_half_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 1 -ah 1          -ap 1          -dbi "overlap" -wi "own"   -il "label_half_reserved"  -o "vitro_e/vivo+vitro/R1/S/repeat_own_label_half_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 1 -ah 1          -ap 1          -dbi "overlap" -wi "own"   -il "both_half_reserved"   -o "vitro_e/vivo+vitro/R1/S/repeat_own_both_half_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 1 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "both_half_reserved"   -o "vitro_e/vivo+vitro/R1/S/repeat_both_half_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -n 1 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "both_reserved"        -o "vitro_e/vivo+vitro/R1/S/repeat_both_reserved.txt"


# waiting:
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invitro/invitro_merged.csv -iv both -n 1  -o vitro(t&e)/s_rasar_bi.txt
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invitro/invitro_both_repeated.csv -iv both -n 1  -o vitro(t&e)/s_rasar_bi_repeated.txt
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/TOXCAST/toxcast_zebrafish_selected.csv -iv toxcast -n 1  -o vitro_t/s_rasar_bi.txt
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/TOXCAST/toxcast_repeated.csv -iv toxcast -n 1  -o vitro_t/s_rasar_bi_repeated.txt
#


# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invitro/invitro_merged.csv         -iv both    -n 4  -o "vitro(t&e)/s_rasar_bi_4.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invitro/invitro_both_repeated.csv  -iv both    -n 4  -o "vitro(t&e)/s_rasar_bi_repeated_4.txt"

# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/TOXCAST/toxcast_processed.csv      -iv toxcast -n 4  -o "vitro_t/s_rasar_bi_4.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/TOXCAST/toxcast_repeated.csv       -iv toxcast -n 4  -o "vitro_t/s_rasar_bi_repeated_4.txt"


# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_toxcast.csv -n 4 -ah "logspace" -ap "logspace" -dbi "no"      -wi "False" -il "both"        -o "vitro_t/vivo+vitro/bestR/repeat.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_toxcast.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "own"   -il "number"      -o "vitro_t/vivo+vitro/bestR/repeat_own_number.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_toxcast.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "own"   -il "label"       -o "vitro_t/vivo+vitro/bestR/repeat_own_label.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_toxcast.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "own"   -il "label_half"  -o "vitro_t/vivo+vitro/bestR/repeat_own_label_half.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_toxcast.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "number"      -o "vitro_t/vivo+vitro/bestR/repeat_number.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_toxcast.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "label"       -o "vitro_t/vivo+vitro/bestR/repeat_label.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_toxcast.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "label_half"  -o "vitro_t/vivo+vitro/bestR/repeat_label_half.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_toxcast.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "both_half"   -o "vitro_t/vivo+vitro/bestR/repeat_both_half.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_toxcast.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "both"        -o "vitro_t/vivo+vitro/bestR/repeat_both.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_toxcast.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "own"   -il "both"        -o "vitro_t/vivo+vitro/bestR/repeat_own_both.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_toxcast.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "own"   -il "both_half"   -o "vitro_t/vivo+vitro/bestR/repeat_own_both_half.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro.csv         -n 4 -ah "logspace" -ap "logspace" -dbi "no"      -wi "False" -il "both"        -o "vitro(t&e)/vivo+vitro/bestR/repeat.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro.csv         -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "own"   -il "number"      -o "vitro(t&e)/vivo+vitro/bestR/repeat_own_number.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro.csv         -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "own"   -il "label"       -o "vitro(t&e)/vivo+vitro/bestR/repeat_own_label.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro.csv         -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "own"   -il "label_half"  -o "vitro(t&e)/vivo+vitro/bestR/repeat_own_label_half.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro.csv         -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "label_half"  -o "vitro(t&e)/vivo+vitro/bestR/repeat_label_half.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro.csv         -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "number"      -o "vitro(t&e)/vivo+vitro/bestR/repeat_number.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro.csv         -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "label"       -o "vitro(t&e)/vivo+vitro/bestR/repeat_label.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro.csv         -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "both_half"   -o "vitro(t&e)/vivo+vitro/bestR/repeat_both_half.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro.csv         -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "own"   -il "both_half"   -o "vitro(t&e)/vivo+vitro/bestR/repeat_own_both_half.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro.csv         -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "own"   -il "both"        -o "vitro(t&e)/vivo+vitro/bestR/repeat_own_both.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro.csv         -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "both"        -o "vitro(t&e)/vivo+vitro/bestR/repeat_both.txt"


# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "own"   -il "both_reserved"        -o "vitro(t&e)/vivo+vitro/bestR/repeat_own_both_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "own"   -il "label_reserved"       -o "vitro(t&e)/vivo+vitro/bestR/repeat_own_label_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "label_reserved"       -o "vitro(t&e)/vivo+vitro/bestR/repeat_label_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "label_half_reserved"  -o "vitro(t&e)/vivo+vitro/bestR/repeat_label_half_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "own"   -il "label_half_reserved"  -o "vitro(t&e)/vivo+vitro/bestR/repeat_own_label_half_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "own"   -il "both_half_reserved"   -o "vitro(t&e)/vivo+vitro/bestR/repeat_own_both_half_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "both_half_reserved"   -o "vitro(t&e)/vivo+vitro/bestR/repeat_both_half_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "both_reserved"        -o "vitro(t&e)/vivo+vitro/bestR/repeat_both_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_toxcast.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "own"   -il "both_reserved"        -o "vitro_t/vivo+vitro/bestR/repeat_own_both_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_toxcast.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "own"   -il "label_reserved"       -o "vitro_t/vivo+vitro/bestR/repeat_own_label_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_toxcast.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "label_reserved"       -o "vitro_t/vivo+vitro/bestR/repeat_label_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_toxcast.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "label_half_reserved"  -o "vitro_t/vivo+vitro/bestR/repeat_label_half_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_toxcast.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "own"   -il "label_half_reserved"  -o "vitro_t/vivo+vitro/bestR/repeat_own_label_half_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_toxcast.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "own"   -il "both_half_reserved"   -o "vitro_t/vivo+vitro/bestR/repeat_own_both_half_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_toxcast.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "both_half_reserved"   -o "vitro_t/vivo+vitro/bestR/repeat_both_half_reserved.txt"
# python RASAR_mulneigh_bi.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_toxcast.csv -n 4 -ah "logspace" -ap "logspace" -dbi "overlap" -wi "True"  -il "both_reserved"        -o "vitro_t/vivo+vitro/bestR/repeat_both_reserved.txt"
