import os
import pickle
import argparse
import pandas as pd
from collections import defaultdict
from utils import save_kg
from knowledge_graph import KnowledgeGraph


def generate_labels(data_dir, output_dir, mode='train'):
    data = pd.read_csv(f"{data_dir}/{mode}.csv")
    data[['source_local_id', 'target_local_id']] = data[['source_local_id', 'target_local_id']].astype(int)
    data_group = data.groupby('source_local_id')
    source_target_dict = defaultdict(list)
    for source_local_id, group in data_group:
        source_target_dict[source_local_id] = list(group['target_local_id'])
    with open(f"{output_dir}/{mode}.pkl", 'wb') as f:
        pickle.dump(source_target_dict, f)
    return source_target_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/linjie/projects/KG/PGPR_hetionet/data', help='data directory')
    parser.add_argument('--dataset', type=str, default='hetionet', help='dataset name')
    parser.add_argument('--output_dir', type=str, default='/home/linjie/projects/KG/PGPR_hetionet/output', help='output directory')
    args = parser.parse_args()
    args.data_dir = args.data_path +'/'+ args.dataset

    # Create hetionet knowledge graph instance.
    # ========== BEGIN ========== #
    kg = KnowledgeGraph(args.data_dir)
    kg.compute_degrees()
    save_kg(args.output_dir, kg)
    # =========== END =========== #

    # Genereate train/test labels.
    # # ========== BEGIN ========== #
    # print('Generate', args.dataset, 'train/test labels.')
    # generate_labels(args.data_dir, args.output_dir, mode='train')
    # generate_labels(args.data_dir, args.output_dir, mode='test')
    # =========== END =========== #


if __name__ == '__main__':
    main()

