import pandas as pd

import numpy as np
from time import ctime
import argparse
from helper_cte_model import *

from sklearn.model_selection import train_test_split, ParameterSampler

from pebble import ProcessPool
from concurrent.futures import TimeoutError
import traceback
import datetime

import logging

# this file are used to test the rnighbor number's effect in simple RASAR function. also the in vitro \
# threshold is studied. different straitifed way are also been looked.

# sys.modules["__main__"].__file__ = "ipython"


def getArguments():
    parser = argparse.ArgumentParser(description="Running T_models for datasets.")
    parser.add_argument(
        "-i", "--input", dest="inputFile", help="input file position", required=True
    )
    parser.add_argument(
        "-iv",
        "--input_vitro",
        dest="inputFile_vitro",
        help="input invitro file position",
        default="no",
    )
    parser.add_argument("-ah", "--alpha_h", dest="hamming_alpha", default="logspace")
    parser.add_argument("-ap", "--alpha_p", dest="pubchem2d_alpha", default="logspace")
    parser.add_argument(
        "-dbi",
        "--db_invitro",
        dest="db_invitro",
        help="yes: add in vitro as other source for distance matrix, no: do not use in vitro as input, overlap: use in vitro as input feature",
        default="no",
    )
    parser.add_argument(
        "-e",
        "--encoding",
        dest="encoding",
        help="binary or multiclass (5 class)",
        default="binary",
    )
    parser.add_argument(
        "-il",
        "--invitro_label",
        dest="invitro_label",
        help="number, label",
        default="number",
    )
    parser.add_argument(
        "-m",
        "--model_type",
        help="model: logistic regression(LR), random forest(RF),decision tree(DT)",
        default="RF",
    )
    parser.add_argument(
        "-n",
        "--neighbors",
        dest="neighbors",
        required=True,
        help="cloest neighbor number in rasar function",
        nargs="+",
        type=int,
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
        "-r",
        "--repeat",
        dest="repeat",
        help="repeat time for different splitting method",
        default=20,
        type=int,
    )
    parser.add_argument("-t_ls", "--t_ls", dest="t_ls", default="median")
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


def s_rasar_func(
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
    matrices_invitro,
    db_invitro,
    max_euc,
    num,
    model,
):

    if args.w_invitro == "own":
        train_rf = pd.DataFrame()
        test_rf = pd.DataFrame()
    else:
        train_rf, test_rf = cal_s_rasar(
            dist_matr_train, dist_matr_test, y_train, n_neighbors, args.encoding,
        )

    if args.w_invitro != "False":
        if str(args.db_invitro) == "overlap":
            train_rf = get_vitroinfo(train_rf, X, train_index, args.invitro_label)
            test_rf = get_vitroinfo(test_rf, X, test_index, args.invitro_label)
        else:
            db_invitro_matrix_normed = pd.DataFrame(
                ah * matrices_invitro["hamming"]
                + ap * matrices_invitro["pubchem"]
                + pd.DataFrame(matrices_invitro["euc"]).divide(max_euc).values
            )
            train_rf = find_nearest_vitro(
                train_rf,
                db_invitro,
                db_invitro_matrix_normed,
                train_index,
                args.invitro_label,
            )
            test_rf = find_nearest_vitro(
                test_rf,
                db_invitro,
                db_invitro_matrix_normed,
                test_index,
                args.invitro_label,
            )

    df_score = fit_and_predict(model, train_rf, y_train, test_rf, y_test, args.encoding)

    df_score["neighbors"] = n_neighbors
    df_score["ah"] = ah
    df_score["ap"] = ap
    df_score["fold"] = num
    return df_score


if __name__ == "__main__":
    conc_column = "conc1_mean"

    if args.encoding == "binary":
        encoding = "binary"
        encoding_value = 1
    elif args.encoding == "multiclass":
        encoding = "multiclass"
        encoding_value = [0.1, 1, 10, 100]

    # loading data & splitting into train and test dataset
    print("loading dataset...", ctime())
    if args.db_invitro == "no" or args.db_invitro == "overlap":
        db_mortality = load_data(
            args.inputFile,
            encoding=encoding,
            categorical_columns=categorical,
            conc_column=conc_column,
            encoding_value=encoding_value,
            seed=1,
        )
        db_invitro = args.db_invitro
    else:
        db_mortality, db_invitro = load_invivo_invitro(
            args.inputFile,
            args.inputFile_vitro,
            encoding=encoding,
            encoding_value=encoding_value,
            seed=42,
        )

    df_fishchem = db_mortality[["fish", "test_cas", conc_column]]
    X = db_mortality.drop(columns=conc_column)
    print("finish loaded.", ctime())

    idx_record = []
    for repeat in range(args.repeat):

        grid_search = pd.DataFrame()
        logging.basicConfig(
            filename=args.outputFile + "_log_{}.log".format(repeat),
            level=logging.DEBUG,
            format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
        )
        logging.info("The" + str(repeat) + "th times:")

        # ----------------------train valid dataset splitting-----------------------
        traintest_idx, valid_idx = get_grouped_train_test_split(
            df_fishchem, test_size=0.2, col_groups="test_cas", rand=repeat,
        )

        if list(valid_idx) in idx_record:
            continue
        else:
            idx_record.append(list(valid_idx))

        X_traintest, X_valid, Y_traintest, Y_valid = get_train_test_data(
            db_mortality, traintest_idx, valid_idx, conc_column
        )

        # -----------------creating the distance matrix--------
        print("calcultaing distance matrix..", ctime())

        matrices = cal_matrixs(X_traintest, X_traintest, categorical, non_categorical)

        matrices_full = cal_matrixs(X, X, categorical, non_categorical)

        if args.db_invitro != "no" and args.db_invitro != "overlap":
            matrices_invitro = cal_matrixs(
                X_traintest, db_invitro, categorical_both, non_categorical
            )
            matrices_invitro_full = cal_matrixs(
                X, db_invitro, categorical_both, non_categorical
            )
        else:
            matrices_invitro = None

        print("distance matrix calculation finished", ctime())

        # -------------------hyperparameter range, alphas, thresholds------------

        # alphas
        if args.hamming_alpha == "logspace":
            sequence_ah = np.logspace(-2, 0, 30)
        else:
            sequence_ah = [float(args.hamming_alpha)]

        if args.pubchem2d_alpha == "logspace":
            sequence_ap = np.logspace(-2, 0, 30)
        else:
            sequence_ap = [float(args.pubchem2d_alpha)]

        # in vitro threshold
        if args.t_ls == "median":
            threshold_ls = X_traintest["invitro_conc"].quantile([0.5]).values
        elif args.t_ls == "best":
            threshold_ls = [
                pd.read_csv(
                    "/local/wujimeng/code_jimeng/c_cte/GitHub/T_models/T_null/repeat_own_label_"
                    + str(repeat)
                    + ".txt"
                )
                .sort_values(
                    by=["train_dataset_accuracy", "test_dataset_accuracy"],
                    ascending=False,
                )
                .threshold.values[0]
            ]
        else:
            threshold_ls = np.logspace(-1, 0.9, 20)  # could be changed
            # threshold_ls = np.logspace(-5, 11, 50)
        print(threshold_ls)

        # models parameters
        hyper_params_tune, model = set_hyperparameters(args.model_type)

        params_comb = list(
            ParameterSampler(hyper_params_tune, n_iter=args.niter, random_state=2,)
        )
        # -------------------training using the default model setting and found the best alphas combination--------------------

        group_kfold = GroupKFold(n_splits=5)

        df_fishchem_tv = df_fishchem.iloc[traintest_idx, :]
        folds = list(
            group_kfold.split(df_fishchem_tv, groups=df_fishchem_tv["test_cas"])
        )

        count = 1
        starttime = datetime.datetime.now()
        num_runs = (
            len(sequence_ap)
            * len(sequence_ah)
            * len(args.neighbors)
            * len(threshold_ls)
            + args.niter
        )

        # ---------------get the best combination of alphas and threholds---------
        best_accs = 0
        grid_search = pd.DataFrame()

        for nei in args.neighbors:
            for ah in sequence_ah:
                for ap in sequence_ap:
                    for thres in threshold_ls:
                        X_traintest = vitroconc_to_label(X_traintest, thres)
                        X_valid = vitroconc_to_label(X_valid, thres)
                        X_labled = vitroconc_to_label(X, thres)
                        if args.db_invitro == "yes":
                            db_invitro = vitroconc_to_label(db_invitro, thres)

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
                                thres,
                                "*" * 50,
                                count / num_runs,
                                remain_time,
                                "remained.",
                                ctime(),
                                end="\r",
                            )
                            count = count + 1
                            results = []

                            with ProcessPool(max_workers=5) as pool:
                                for num, fold in enumerate(folds):
                                    y_train = Y_traintest[fold[0]]
                                    y_test = Y_traintest[fold[1]]

                                    matrix_euc = pd.DataFrame(matrices["euc"])
                                    max_euc = matrix_euc.iloc[
                                        fold[0], fold[0]
                                    ].values.max()

                                    if args.w_invitro == "own":
                                        train_matrix = pd.DataFrame()
                                        test_matrix = pd.DataFrame()
                                    else:
                                        (
                                            train_matrix,
                                            test_matrix,
                                        ) = get_traintest_matrices(
                                            matrices, fold, ah, ap
                                        )
                                    res = pool.schedule(
                                        s_rasar_func,
                                        args=(
                                            fold[0],
                                            fold[1],
                                            train_matrix,
                                            test_matrix,
                                            y_train,
                                            y_test,
                                            X_traintest,
                                            nei,
                                            ah,
                                            ap,
                                            matrices_invitro,
                                            db_invitro,
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

                                results = [res.result() for res in results]

                            result_stats = result_describe(results)

                            # ---------------testing on the validation dataset---------
                            traintest_idx = X_traintest.index
                            valid_idx = X_valid.index

                            if args.w_invitro == "own":
                                train_rf = pd.DataFrame()
                                test_rf = pd.DataFrame()
                            else:

                                matrix_trainvalid, matrix_test = get_traintest_matrices(
                                    matrices_full, [traintest_idx, valid_idx], ah, ap
                                )

                                train_rf, test_rf = cal_s_rasar(
                                    matrix_trainvalid,
                                    matrix_test,
                                    Y_traintest,
                                    int(nei),
                                    encoding,
                                )

                            if args.w_invitro != "False":
                                if str(args.db_invitro) == "overlap":
                                    train_rf = get_vitroinfo(
                                        train_rf,
                                        X_labled,
                                        traintest_idx,
                                        args.invitro_label,
                                    )
                                    test_rf = get_vitroinfo(
                                        test_rf,
                                        X_labled,
                                        valid_idx,
                                        args.invitro_label,
                                    )
                                else:

                                    matrices_invitro_full_norm = normalized_invitro_matrix(
                                        X_traintest, matrices_invitro_full, ah, ap,
                                    )

                                    train_rf = find_nearest_vitro(
                                        train_rf,
                                        db_invitro,
                                        matrices_invitro_full_norm,
                                        traintest_idx,
                                        args.invitro_label,
                                    )
                                    test_rf = find_nearest_vitro(
                                        test_rf,
                                        db_invitro,
                                        matrices_invitro_full_norm,
                                        valid_idx,
                                        args.invitro_label,
                                    )

                            df_test_score = fit_and_predict(
                                model,
                                train_rf,
                                Y_traintest,
                                test_rf,
                                Y_valid,
                                args.encoding,
                            )

                            try:
                                dict_acc = dataset_acc(
                                    X_traintest, X_valid, Y_traintest, Y_valid,
                                )
                            except:
                                dict_acc = pd.DataFrame()

                            temp_grid = save_results(
                                "default",
                                ah,
                                ap,
                                result_stats,
                                df_test_score,
                                dict_acc,
                            )
                            temp_grid["threshold"] = int(thres)
                            grid_search = pd.concat([grid_search, temp_grid])

                            if result_stats[0].accuracy.values[0] > best_accs:
                                best_accs = result_stats[0].accuracy.values[0]
                                best_ah = ah
                                best_ap = ap
                                best_thres = thres
                                best_score = temp_grid
                                print(
                                    "success found better alphas and threshold",
                                    best_accs,
                                    end="\r",
                                )
                        except ValueError as error:
                            logging.error(str(error))

            del results
        # ---------------------------------------------

        # --------------explore more on the model's hyperparameter range with found alphas--------
        print(
            "best alphas and threshold found, ah:{}, ap:{}, threshold: {}, accuracy:{}".format(
                best_ah, best_ap, best_thres, best_accs
            )
        )
        X_traintest = vitroconc_to_label(X_traintest, best_thres)
        X_valid = vitroconc_to_label(X_valid, best_thres)
        X_labled = vitroconc_to_label(X, best_thres)
        if args.db_invitro == "yes":
            db_invitro = vitroconc_to_label(db_invitro, best_thres)

        for j in range(0, len(params_comb)):

            for k, v in params_comb[j].items():
                setattr(model, k, v)

            try:
                remain_time = (
                    (datetime.datetime.now() - starttime) / count * (num_runs - count)
                )
                print(
                    best_ah,
                    ",",
                    best_ap,
                    ",",
                    nei,
                    ",",
                    best_thres,
                    "*" * 50,
                    count / num_runs,
                    remain_time,
                    "remained.",
                    ctime(),
                    end="\r",
                )

                count = count + 1
                results = []
                with ProcessPool(max_workers=5) as pool:
                    for num, fold in enumerate(folds):
                        y_train = Y_traintest[fold[0]]
                        y_test = Y_traintest[fold[1]]

                        matrix_euc = pd.DataFrame(matrix_euc)
                        max_euc = matrix_euc.iloc[fold[0], fold[0]].values.max()

                        if args.w_invitro == "own":
                            train_matrix = pd.DataFrame()
                            test_matrix = pd.DataFrame()
                        else:
                            (train_matrix, test_matrix,) = get_traintest_matrices(
                                matrices, fold, best_ah, best_ap
                            )

                        res = pool.schedule(
                            s_rasar_func,
                            args=(
                                fold[0],
                                fold[1],
                                train_matrix,
                                test_matrix,
                                y_train,
                                y_test,
                                X_traintest,
                                nei,
                                best_ah,
                                best_ap,
                                matrices_invitro,
                                db_invitro,
                                max_euc,
                                num,
                                model,
                            ),
                            timeout=120,
                        )

                        results.append(res)

                        del train_matrix, test_matrix
                    results = [i.result() for i in results]

                result_stats = result_describe(results)

                # ---------------testing on the validation dataset---------
                # --------------save all the runs and the one results with best parameters combination---------------

                if args.w_invitro == "own":
                    train_rf = pd.DataFrame()
                    test_rf = pd.DataFrame()
                else:

                    (matrix_trainvalid, matrix_test,) = get_traintest_matrices(
                        matrices_full, [traintest_idx, valid_idx], best_ah, best_ap
                    )

                    train_rf, test_rf = cal_s_rasar(
                        matrix_trainvalid, matrix_test, Y_traintest, int(nei), encoding,
                    )

                if args.w_invitro != "False":
                    if str(args.db_invitro) == "overlap":
                        train_rf = get_vitroinfo(
                            train_rf, X_labled, traintest_idx, args.invitro_label,
                        )
                        test_rf = get_vitroinfo(
                            test_rf, X_labled, valid_idx, args.invitro_label,
                        )
                    else:

                        # db_invitro_matrix_norm = normalized_invitro_matrix(
                        #     X_traintest, matrices_invitro, best_ah, best_ap
                        # )
                        matrices_invitro_full_norm = normalized_invitro_matrix(
                            X_traintest, matrices_invitro_full, best_ah, best_ap
                        )

                        train_rf = find_nearest_vitro(
                            train_rf,
                            db_invitro,
                            matrices_invitro_full_norm,
                            traintest_idx,
                            args.invitro_label,
                        )
                        test_rf = find_nearest_vitro(
                            test_rf,
                            db_invitro,
                            matrices_invitro_full_norm,
                            valid_idx,
                            args.invitro_label,
                        )

                df_test_score = fit_and_predict(
                    model, train_rf, Y_traintest, test_rf, Y_valid, args.encoding,
                )

                try:
                    dict_acc = dataset_acc(X_traintest, X_valid, Y_traintest, Y_valid)
                except:
                    dict_acc = pd.DataFrame()

                temp_grid = save_results(
                    params_comb[j],
                    best_ah,
                    best_ap,
                    result_stats,
                    df_test_score,
                    dict_acc,
                )

                temp_grid["threshold"] = int(best_thres)

                grid_search = pd.concat([grid_search, temp_grid])

            except ValueError as error:
                logging.error(str(error))

            if result_stats[0].accuracy.values[0] > best_accs:
                best_accs = result_stats[0].accuracy.values[0]
                best_score = temp_grid
                print("success found better hyperparameters", best_accs)

        print("finished", ctime())
        df2file(grid_search, args.outputFile + "_{}_fullinfo.txt".format(repeat))
        df2file(best_score, args.outputFile + "_{}.txt".format(repeat))


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