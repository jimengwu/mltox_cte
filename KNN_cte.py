# from helper_model import *
from helper_cte_model import *
import numpy as np
from time import ctime
import argparse


def getArguments():
    parser = argparse.ArgumentParser(description="Running KNN_model for datasets.")
    parser.add_argument("-i", "--input", dest="inputFile", required=True)
    parser.add_argument("-e", "--encoding", dest="encoding", default="binary")
    parser.add_argument("-l", "--leaf_ls", dest="leaf_list", required=True, nargs="+")
    parser.add_argument(
        "-n", "--neighbors", dest="neighbors", required=True, nargs="+", type=int
    )
    parser.add_argument("-o", "--output", dest="outputFile", default="binary.txt")
    return parser.parse_args()


rand = random.randrange(1, 100)
invitro = True  # set whether the in vitro information is included as input feature
categorical = [
    "class",
    "tax_order",
    "family",
    "genus",
    "species",
    "control_type",
    "media_type",
    "application_freq_unit",
    "exposure_type",
    "conc1_type",
    "obs_duration_mean",
    "invitro_label",
]

if invitro:
    categorical = ["class", "tax_order", "family", "genus", "species"]
# non_categorical was numerical features, which will be standarized. \
# Mol,bonds_number, atom_number was previously log transformed due to the maginitude of their values.

non_categorical = [
    "ring_number",
    "tripleBond",
    "doubleBond",
    "alone_atom_number",
    "oh_count",
    "atom_number",
    "bonds_number",
    "MorganDensity",
    "LogP",
    "mol_weight",
    "water_solubility",
    "melting_point",
    # "invitro_conc"
]
args = getArguments()

if args.encoding == "binary":
    encoding = "binary"
    encoding_value = 1
elif args.encoding == "multiclass":
    encoding = "multiclass"
    encoding_value = [0.1, 1, 10, 100]

# -----------loading data & splitting into train and test dataset--------
print("loading dataset...", ctime())
db_mortality = load_data(
    args.inputFile,
    encoding=encoding,
    categorical_columns=categorical,
    encoding_value=encoding_value,
    seed=42,
)
print("finish loaded.", ctime())

# -----------------startified split the dataset for training model--------
df_fishchem = db_mortality[["fish", "test_cas"]]
test_size = 0.2
col_groups = "test_cas"

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

Y_traintest = db_mortality.iloc[traintest_idx, :].conc1_mean
Y_valid = db_mortality.iloc[valid_idx, :].conc1_mean


# ------using 5-fold cross validation to choose the alphas with best accuracy-----

# sequence_alpha = np.concatenate([
#     np.logspace(-1.5, -1, 20),
# np.logspace(-3, 0, 25),
# np.linspace(1.5, 10, 5),
# np.logspace(-2, 1, 30),
# np.logspace(-0.1, 1, 30),
# np.logspace(-0.8, 0, 10)
# ])
# sequence_alpha = np.logspace(-5, 0, 30)
sequence_alpha = np.logspace(-2, 1, 30)
# sequence_alpha = np.logspace(-2, 1, 3)  # just for testing

best_results = select_alpha(
    df_fishchem_tv,
    col_groups,
    X_traintest,
    Y_traintest,
    sequence_alpha,
    categorical,
    non_categorical,
    args.leaf_list,
    args.neighbors,
    encoding,
)
print("training finished.", ctime())

# ----------------------validate on dataset---------------
minmax = MinMaxScaler().fit(
    X_traintest[non_categorical]
)  # normalized the validation dataset
X_traintest[non_categorical] = minmax.transform(X_traintest.loc[:, non_categorical])
X_valid[non_categorical] = minmax.transform(X_valid.loc[:, non_categorical])


matrix_traintest = dist_matrix(
    X_traintest,
    X_traintest,
    non_categorical,
    categorical,
    best_results.iloc[0]["ah"],
    best_results.iloc[0]["ap"],
)
matrix_valid = dist_matrix(
    X_valid,
    X_traintest,
    non_categorical,
    categorical,
    best_results.iloc[0]["ah"],
    best_results.iloc[0]["ap"],
)

neigh = KNeighborsClassifier(
    n_neighbors=best_results.iloc[0]["neighbor"],
    metric="precomputed",
    leaf_size=best_results.iloc[0]["leaf"],
)

df_output = fit_and_predict(
    neigh,
    matrix_traintest,
    Y_traintest.astype("int").ravel(),
    matrix_valid,
    Y_valid,
    encoding,
)

# --------saving the information into a file

df_output.loc[0, ["ah", "ap", "leaf", "neighbor"]] = best_results.iloc[0][
    ["ah", "ap", "leaf", "neighbor"]
]
df_output.loc[0, "sequence_alpha"] = str(sequence_alpha)
df_output.loc[0, "leaf_list"] = str(args.leaf_list)
df_output.loc[0, "encoding"] = args.encoding

result = pd.concat([df_output, best_results], keys=["valid", "train"])

df2file(result, args.outputFile)


# python KNN_cte.py -i  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv   -l 10 -n 1 -o   knn/1nn_bi.txt
# python KNN_cte.py -i  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv   -l 10 -n 3 -o   knn/3nn_bi.txt
# python KNN_cte.py -i  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv   -l 10 -n 5 -o   knn/5nn_bi.txt


# python KNN_cte.py -i /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -e "multiclass" -l 10 30 50 70 90  -n 1 -o   knn/1nn_mul.txt
# python KNN_cte.py -i /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -e "multiclass" -l 10 30 50 70 90  -n 3 -o   knn/3nn_mul.txt
# python KNN_cte.py -i /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -e "multiclass" -l 10 30 50 70 90  -n 5 -o   knn/5nn_mul.txt

# python KNN_cte.py -i  data/invivo/invivo_repeated_w_invitro.csv -l 10 30 50  -n 1 3 5 -o   invitro/knn.txt
# python KNN_cte.py -i  data/invivo/invivo_repeated_w_invitro.csv   -e "multiclass" -l 10 30 50  -n 1 3 5 -o   invitro/knn_mul.txt

# python KNN_cte.py -i data/invitro/invitro_merged.csv -l 10 30 50 -n 1 3 5 -o knn/invitro_bi.txt
