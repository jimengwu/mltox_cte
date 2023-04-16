from helper_cte_model import *
from sklearn.model_selection import ParameterSampler
from sklearn.ensemble import RandomForestClassifier
import argparse
import os


def getArguments():
    parser = argparse.ArgumentParser(
        description="Running simple RASAR model with strafitied splitting datasets."
    )
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
    parser.add_argument(
        "-ah", "--alpha_h", dest="hamming_alpha", required=True, nargs="?"
    )
    parser.add_argument("-ap", "--alpha_p", dest="alpha_p", required=True, nargs="?")
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
    parser.add_argument(
        "-if",
        "--invitroFile",
        dest="invitroFile",
        help="if input is invitroFile",
        default=False,
    )
    parser.add_argument(
        "-il",
        "--invitro_label",
        dest="invitro_label",
        help="number, label",
        default="number",
    )
    parser.add_argument(
        "-ivt",
        "--invitro_type",
        dest="invitro_type",
        help="invitro file source: eawag, toxcast, both",
        default="False",
    )
    parser.add_argument(
        "-m",
        "--model_type",
        help="model: logistic regression(LR), random forest(RF),decision tree(DT)",
        default="RF",
    )
    parser.add_argument(
        "-n", "--n_neighbors", dest="n_neighbors", nargs="?", default=1, type=int
    )
    parser.add_argument(
        "-ni",
        "--niter",
        help="number of iterations to find the best parameter combination",
        nargs="?",
        default=100,
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
if args.encoding == "binary":
    encoding = "binary"
    encoding_value = 1
elif args.encoding == "multiclass":
    encoding = "multiclass"
    encoding_value = [0.1, 1, 10, 100]


categorical, conc_column = get_col_ls(args.invitro_type)

rand = random.randrange(1, 100)


# -----------loading data & splitting into train and test dataset--------
print("loading dataset...", ctime())
if args.db_invitro == "no" or args.db_invitro == "overlap":
    db_mortality = load_data(
        args.inputFile,
        encoding=encoding,
        categorical_columns=categorical,
        conc_column=conc_column,
        encoding_value=encoding_value,
        seed=rand,
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
print("finish loaded.", ctime())

# -----------------startified split the dataset for training model--------
test_size = 0.2
col_groups = "test_cas"

df_fishchem = db_mortality[["fish", "test_cas"]]
traintest_idx, valid_idx = get_grouped_train_test_split(
    df_fishchem, test_size, col_groups
)

df_fishchem_tv = df_fishchem.iloc[traintest_idx, :]
X_traintest, X_valid, Y_traintest, Y_valid = get_train_test_data(
    db_mortality, traintest_idx, valid_idx, conc_column
)

if args.db_invitro != "no" and args.db_invitro != "overlap":
    # Encoding for invitro variable: binary and multiclass
    if args.encoding == "binary":
        db_invitro["invitro_label"] = np.where(
            db_invitro["invitro_conc"].values > 1, 0, 1
        )
        db_invitro["invitro_label_half"] = np.where(
            db_invitro["invitro_conc"].median() > 1, 0, 1
        )
    elif args.encoding == "multiclass":
        db_invitro["invitro_label"] = multiclass_encoding(
            db_invitro["invitro_conc"], [0.006, 0.3, 63, 398]
        )


# -----------------creating the distance matrix--------

print("calcultaing distance matrix..", ctime())
matrices = cal_matrixs(X_traintest, X_traintest, categorical, non_categorical)

X = db_mortality.drop(columns=conc_column)
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

# ------------------------hyperparameters range---------
if args.hamming_alpha == "logspace":
    sequence_ah = np.logspace(-2, 0, 30)
else:
    sequence_ah = [float(args.hamming_alpha)]

if args.pubchem2d_alpha == "logspace":
    sequence_ap = np.logspace(-2, 0, 30)
else:
    sequence_ap = [float(args.pubchem2d_alpha)]


hyper_params_tune, model = set_hyperparameters(args.model_type)

params_comb = list(
    ParameterSampler(hyper_params_tune, n_iter=args.niter, random_state=rand)
)

# -------------------training using the default model setting and found the best alphas combination--------------------

best_accs = 0

count = 1
for ah in sequence_ah:
    for ap in sequence_ap:
        for i in range(0, len(params_comb)):
            print(
                "*" * 50,
                count / (len(sequence_ap) * len(sequence_ah) * len(params_comb)),
                ctime(),
                end="\r",
            )
            count = count + 1

            for k, v in params_comb[i].items():
                setattr(model, k, v)

            results = RASAR_simple(
                df_fishchem_tv,
                col_groups,
                matrices,
                ah,
                ap,
                X_traintest,
                Y_traintest,
                db_invitro_matrices=matrices_invitro,
                n_neighbors=args.n_neighbors,
                invitro=args.w_invitro,
                invitro_form=args.invitro_label,
                db_invitro=db_invitro,
                encoding=encoding,
                model=model,
            )

            if results["accuracy"].mean() > best_accs:
                best_accs = results["accuracy"].mean()

                best_results = results

                best_ah = ah

                best_ap = ap

                best_param = params_comb[i]

                print("success found better alphas and hyperparameters", best_accs)
df_mean = pd.DataFrame(best_results.mean(axis=0)).transpose()
df_std = pd.DataFrame(best_results.sem(axis=0)).transpose()
df_std["neighbors"] = args.n_neighbors


# -------------------tested on test dataset--------------------
for k, v in best_param.items():
    setattr(model, k, v)


if args.w_invitro == "own":
    traintest_rf = pd.DataFrame()
    valid_rf = pd.DataFrame()
else:
    (matrix_traintest, matrix_valid) = get_traintest_matrices(
        matrices_full, [traintest_idx, valid_idx], best_ah, best_ap
    )

    traintest_rf, valid_rf = cal_s_rasar(
        matrix_traintest, matrix_valid, Y_traintest, args.n_neighbors, encoding,
    )
if args.w_invitro != "False":
    if str(db_invitro) == "overlap":
        traintest_rf = get_vitroinfo(traintest_rf, X, traintest_idx, args.invitro_label)
        valid_rf = get_vitroinfo(valid_rf, X, valid_idx, args.invitro_label)
    else:

        matrices_invitro_full_norm = normalized_invitro_matrix(
            X_traintest, matrices_invitro_full, best_ah, best_ap
        )

        traintest_rf = find_nearest_vitro(
            traintest_rf,
            db_invitro,
            matrices_invitro_full_norm,
            traintest_idx,
            args.invitro_label,
        )
        valid_rf = find_nearest_vitro(
            valid_rf,
            db_invitro,
            matrices_invitro_full_norm,
            valid_idx,
            args.invitro_label,
        )

df_test_score = fit_and_predict(
    model, traintest_rf, Y_traintest, valid_rf, Y_valid, encoding
)

df_test_score.loc[0, ["ah", "ap"]] = best_ah, best_ap

for k, v in best_param.items():
    df_test_score.loc[0, k] = v

df_test_score.loc[0, "encoding"] = args.encoding


df_output = pd.concat(
    [df_mean, df_std, df_test_score],
    keys=["train_mean", "train_std", "test"],
    names=["series_name"],
)

# --------saving the information into a file
print(df_output)
df2file(df_output, args.outputFile + ".txt")

