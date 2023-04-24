import os
import sys
import json
import argparse
from math import log
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import numpy as np
import gzip
import pickle
import random
from datetime import datetime
import matplotlib.pyplot as plt
import torch


class KnowledgeGraph(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.kg_relation = json.load(open(self.data_dir + "/kg_relation.json", 'r'))
        self.G = dict()
        self.load_entities()
        self.load_knowledge()
        self.clean()
        self.top_matches = None

    def load_entities(self):
        print('Load entities...')
        num_nodes = 0
        entities_df = pd.read_csv(self.data_dir+"/entities_df.csv")
        entities_size = entities_df.groupby("metanode")["entities_value"].count()
        metanode_df = pd.read_csv(self.data_dir+"/metanode.csv")
        for metanode in metanode_df['metanode']:
            self.G[metanode] = {}
            for eid in range(entities_size[metanode]):
                self.G[metanode][eid] = {r: [] for r in self.kg_relation[metanode].keys()}
            num_nodes += entities_size[metanode]
        print('Total {:d} nodes.'.format(num_nodes))

    def load_knowledge(self):
        df_triple = pd.read_csv(self.data_dir+"/df_triples_CmC_NotIn.csv", low_memory=False)
        df_triple[['source_local_id', 'target_local_id']] = df_triple[['source_local_id', 'target_local_id']].astype(int)
        triple_group = df_triple.groupby('metaedge')
        num_edges = {}
        for metaedge, group in tqdm(triple_group, total=len(triple_group)):
            print(f"Load knowledge {metaedge}...")
            num_edges[metaedge] = 0
            for idx, row in group.iterrows():
                self.G[row['source_metanode']][row['source_local_id']][metaedge].append(row['target_local_id'])
                self.G[row['target_metanode']][row['target_local_id']]['_'+metaedge].append(row['source_local_id'])
                num_edges[metaedge] += 2
        print(num_edges)

    def clean(self):
        print('Remove duplicates...')
        for metanode in self.G:
            for eid in self.G[metanode]:
                for r in self.G[metanode][eid]:
                    data = self.G[metanode][eid][r]
                    data = tuple(sorted(set(data)))
                    self.G[metanode][eid][r] = data

    def compute_degrees(self):
        print('Compute node degrees...')
        self.degrees = {}
        self.max_degree = {}
        for metanode in self.G:
            self.degrees[metanode] = {}
            for eid in self.G[metanode]:
                count = 0
                for r in self.G[metanode][eid]:
                    count += len(self.G[metanode][eid][r])
                self.degrees[metanode][eid] = count

    def get(self, eh_type, eh_id=None, relation=None):
        data = self.G
        if eh_type is not None:
            data = data[eh_type]
        if eh_id is not None:
            data = data[eh_id]
        if relation is not None:
            data = data[relation]
        return data

    def __call__(self, eh_type, eh_id=None, relation=None):
        return self.get(eh_type, eh_id, relation)

    def get_tails(self, metanode, entity_id, relation):
        return self.G[metanode][entity_id][relation]

    def get_tails_given_user(self, metanode, entity_id, relation, user_id):
        """ Very important!
        :param metanode:
        :param entity_id:
        :param relation:
        :param user_id:
        :return:
        """
        tail_type = self.kg_relation[metanode][relation]
        tail_ids = self.G[metanode][entity_id][relation]
        if tail_type not in self.top_matches:
            return tail_ids
        top_match_set = set(self.top_matches[tail_type][user_id])
        top_k = len(top_match_set)
        if len(tail_ids) > top_k:
            tail_ids = top_match_set.intersection(tail_ids)
        return list(tail_ids)

    def trim_edges(self):
        degrees = {}
        for entity in self.G:
            degrees[entity] = {}
            for eid in self.G[entity]:
                for r in self.G[entity][eid]:
                    if r not in degrees[entity]:
                        degrees[entity][r] = []
                    degrees[entity][r].append(len(self.G[entity][eid][r]))

        for entity in degrees:
            for r in degrees[entity]:
                tmp = sorted(degrees[entity][r], reverse=True)
                print(entity, r, tmp[:10])



