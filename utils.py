import sys
import pickle
import logging
import numpy as np
import pandas as pd
import torch
import random
import logging.handlers
from collections import defaultdict


def save_kg(output_dir, kg):
    kg_file = output_dir + '/kg.pkl'
    pickle.dump(kg, open(kg_file, 'wb'))


def load_kg(data_dir):
    kg_file = data_dir + '/kg.pkl'
    kg = pickle.load(open(kg_file, 'rb'))
    return kg

def load_embed(data_dir):
    embed_file = f"{data_dir}/transe_embed.pkl"
    print('Load embedding:', embed_file)
    embed = pickle.load(open(embed_file, 'rb'))
    return embed


def get_logger(logname):
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]  %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_labels(data_dir, mode='train'):
    label_file = f"{data_dir}/{mode}.csv"
    df = pd.read_csv(label_file)
    df = df[['source_local_id', 'target_local_id']].astype(int)
    df_group = df.groupby('source_local_id')
    label_dict = defaultdict(list)
    for source_local_id, group in df_group:
        label_dict[source_local_id] = list(group['target_local_id'])
    return label_dict


# metapaths = {
#     1: ((None, "Compount"), ("CtD", "Disease"), ("CtD", "Compount")),
#     2: ((None, "Compount"), ("CpD", "Disease"), ("CpD", "Compount")),
#     3: ((None, "Compount"), ("CtD", "Disease"), ("CpD", "Compount")),
#     4: ((None, "Compount"), ("CtD", "Disease"), ("DaG", "Gene"), ("CdG", "Compount")),
#     5: ((None, "Compount"), ("CtD", "Disease"), ("DaG", "Gene"), ("CuG", "Compount")),
#     6: ((None, "Compount"), ("CtD", "Disease"), ("DaG", "Gene"), ("CbG", "Compount")),
#     7: ((None, "Compount"), ("CrC", "Compount"), ("CrC", "Compount")),
#     8: ((None, "Compount"), ("CbG", "Gene"), ("CbG", "Compount")),
#     9: ((None, "Compount"), ("CuG", "Gene"), ("CuG", "Compount")),
#     10: ((None, "Compount"), ("CdG", "Gene"), ("CdG", "Compount")),
#     11: ((None, "Compount"), ("CbG", "Gene"), ("GiG", "Gene"), ("CbG", "Compount")),
#     12: ((None, "Compount"), ("CbG", "Gene"), ("Gr>G", "Gene"), ("CbG", "Compount")),
#     13: ((None, "Compount"), ("CbG", "Gene"), ("Gr>G", "Gene"), ("CuG", "Compount")),
#     14: ((None, "Compount"), ("CbG", "Gene"), ("Gr>G", "Gene"), ("CdG", "Compount")),
#     15: ((None, "Compount"), ("CbG", "Gene"), ("GpPW", "Pathway"), ("GpPW", "Gene"), ("CbG", "Compount")),
#     16: ((None, "Compount"), ("CdG", "Gene"), ("GpPW", "Pathway"), ("GpPW", "Gene"), ("CdG", "Compount")),
#     17: ((None, "Compount"), ("CuG", "Gene"), ("GpPW", "Pathway"), ("GpPW", "Gene"), ("CuG", "Compount")),
#     15: ((None, "Compount"), ("CbG", "Gene"), ("GpMF", "Pathway"), ("GpMF", "Gene"), ("CbG", "Compount")),
#     16: ((None, "Compount"), ("CdG", "Gene"), ("GpMF", "Pathway"), ("GpMF", "Gene"), ("CdG", "Compount")),
#     17: ((None, "Compount"), ("CuG", "Gene"), ("GpMF", "Pathway"), ("GpMF", "Gene"), ("CuG", "Compount")),
#     18: ((None, "Compount"), ("CbG", "Gene"), ("GpCC", "CellularComponent"), ("GpCC", "Gene"), ("CbG", "Compount")),
#     19: ((None, "Compount"), ("CdG", "Gene"), ("GpCC", "CellularComponent"), ("GpCC", "Gene"), ("CdG", "Compount")),
#     20: ((None, "Compount"), ("CuG", "Gene"), ("GpCC", "CellularComponent"), ("GpCC", "Gene"), ("CuG", "Compount")),
#     21: ((None, "Compount"), ("CbG", "Gene"), ("GpBP", "BiologicalProcess"), ("GpBP", "Gene"), ("CbG", "Compount")),
#     22: ((None, "Compount"), ("CdG", "Gene"), ("GpBP", "BiologicalProcess"), ("GpBP", "Gene"), ("CdG", "Compount")),
#     23: ((None, "Compount"), ("CuG", "Gene"), ("GpBP", "BiologicalProcess"), ("GpBP", "Gene"), ("CuG", "Compount")),
# }



metapaths = {
    1: ((None, "Compount"), ("CtD", "Disease"), ("_CtD", "Compount")),
    2: ((None, "Compount"), ("CpD", "Disease"), ("_CpD", "Compount")),
    3: ((None, "Compount"), ("CtD", "Disease"), ("_CpD", "Compount")),
    4: ((None, "Compount"), ("CtD", "Disease"), ("DaG", "Gene"), ("_CdG", "Compount")),
    5: ((None, "Compount"), ("CtD", "Disease"), ("DaG", "Gene"), ("_CuG", "Compount")),
    6: ((None, "Compount"), ("CtD", "Disease"), ("DaG", "Gene"), ("_CbG", "Compount")),
    7: ((None, "Compount"), ("CrC", "Compount"), ("CrC", "Compount")),
    8: ((None, "Compount"), ("CrC", "Compount"), ("_CrC", "Compount")),
    9: ((None, "Compount"), ("CbG", "Gene"), ("_CbG", "Compount")),
    10: ((None, "Compount"), ("CuG", "Gene"), ("_CuG", "Compount")),
    11: ((None, "Compount"), ("CdG", "Gene"), ("_CdG", "Compount")),
    12: ((None, "Compount"), ("CbG", "Gene"), ("GiG", "Gene"), ("_CbG", "Compount")),
    13: ((None, "Compount"), ("CbG", "Gene"), ("_GiG", "Gene"), ("_CbG", "Compount")),
    14: ((None, "Compount"), ("CbG", "Gene"), ("Gr>G", "Gene"), ("_CbG", "Compount")),
    15: ((None, "Compount"), ("CbG", "Gene"), ("Gr>G", "Gene"), ("_CuG", "Compount")),
    16: ((None, "Compount"), ("CbG", "Gene"), ("Gr>G", "Gene"), ("_CdG", "Compount")),
    17: ((None, "Compount"), ("CbG", "Gene"), ("GpPW", "Pathway"), ("_GpPW", "Gene"), ("_CbG", "Compount")),
    18: ((None, "Compount"), ("CdG", "Gene"), ("GpPW", "Pathway"), ("_GpPW", "Gene"), ("_CdG", "Compount")),
    19: ((None, "Compount"), ("CuG", "Gene"), ("GpPW", "Pathway"), ("_GpPW", "Gene"), ("_CuG", "Compount")),
    20: ((None, "Compount"), ("CbG", "Gene"), ("GpMF", "Pathway"), ("_GpMF", "Gene"), ("_CbG", "Compount")),
    21: ((None, "Compount"), ("CdG", "Gene"), ("GpMF", "Pathway"), ("_GpMF", "Gene"), ("_CdG", "Compount")),
    22: ((None, "Compount"), ("CuG", "Gene"), ("GpMF", "Pathway"), ("_GpMF", "Gene"), ("_CuG", "Compount")),
    23: ((None, "Compount"), ("CbG", "Gene"), ("GpCC", "CellularComponent"), ("_GpCC", "Gene"), ("_CbG", "Compount")),
    24: ((None, "Compount"), ("CdG", "Gene"), ("GpCC", "CellularComponent"), ("_GpCC", "Gene"), ("_CdG", "Compount")),
    25: ((None, "Compount"), ("CuG", "Gene"), ("GpCC", "CellularComponent"), ("_GpCC", "Gene"), ("_CuG", "Compount")),
    26: ((None, "Compount"), ("CbG", "Gene"), ("GpBP", "BiologicalProcess"), ("_GpBP", "Gene"), ("_CbG", "Compount")),
    27: ((None, "Compount"), ("CdG", "Gene"), ("GpBP", "BiologicalProcess"), ("_GpBP", "Gene"), ("_CdG", "Compount")),
    28: ((None, "Compount"), ("CuG", "Gene"), ("GpBP", "BiologicalProcess"), ("_GpBP", "Gene"), ("_CuG", "Compount")),
}




