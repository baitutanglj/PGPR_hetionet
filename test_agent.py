import os
import json
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from functools import reduce
from kg_env import BatchKGEnvironment
from train_agent import ActorCritic
from utils import load_labels, load_embed

SELF_LOOP = "self_loop"

def metrics_hit(pred_labels, head_tail_dict, output_dir):
    hit1_num, hit3_num, hit10_num, mrr = 0, 0, 0, 0
    for head_entity in head_tail_dict.keys():
        pred_tail_ids = pred_labels[head_entity]
        true_tail_ids = head_tail_dict[head_entity]
        pred_hit_tail_ids = list(set(pred_tail_ids) & set(true_tail_ids))
        if len(pred_hit_tail_ids)>0:
            hit1_num = hit1_num + 1 if len(set(pred_tail_ids[:1])&set(true_tail_ids))>0 else hit1_num
            hit3_num = hit3_num + 1 if len(set(pred_tail_ids[:3]) & set(true_tail_ids)) > 0 else hit3_num
            hit10_num = hit10_num + 1 if len(set(pred_tail_ids[:10]) & set(true_tail_ids)) > 0 else hit10_num
            # tmp = min([pred_tail_ids.index(i) for i in pred_hit_tail_ids]) if len(pred_hit_tail_ids)>0 else 0
            mrr = mrr + 1 / (min([pred_tail_ids.index(i) for i in pred_hit_tail_ids])+1)

    hit1 = hit1_num / len(pred_labels)
    hit3 = hit3_num / len(pred_labels)
    hit10 = hit10_num / len(pred_labels)
    mrr = mrr / len(pred_labels)
    metric_df = pd.DataFrame({'hit1': hit1, 'hit3': hit3, 'hit10': hit10,
                              'hit1_num': hit1_num, 'hit3_num': hit3_num, 'hit10_num': hit10_num,
                              'mrr': mrr}, index=[0]).round(3)
    metric_df.to_csv(output_dir + '/metric_hit.csv', index=False)
    print(metric_df)
    return metric_df

def batch_beam_search(env, model, head_ids, device, kg_relation, topk=[25, 5, 1]):
    def _batch_acts_to_masks(batch_acts):
        batch_masks = []
        for acts in batch_acts:
            num_acts = len(acts)
            act_mask = np.zeros(model.act_dim, dtype=np.uint8)
            act_mask[:num_acts] = 1
            batch_masks.append(act_mask)
        return np.vstack(batch_masks)

    state_pool = env.reset(head_ids)  # numpy of [bs, dim]
    path_pool = env._batch_path  # list of list, size=bs
    probs_pool = [[] for _ in head_ids]
    model.eval()
    for hop in range(3):
        state_tensor = torch.FloatTensor(state_pool).to(device)
        acts_pool = env._batch_get_actions(path_pool, False)  # list of list, size=bs
        actmask_pool = _batch_acts_to_masks(acts_pool)  # numpy of [bs, dim]
        actmask_tensor = torch.ByteTensor(actmask_pool).to(device)
        probs, _ = model((state_tensor, actmask_tensor))  # Tensor of [bs, act_dim]
        probs = probs + actmask_tensor.float()  # In order to differ from masked actions
        topk_probs, topk_idxs = torch.topk(probs, topk[hop], dim=1)  # LongTensor of [bs, k]
        topk_idxs = topk_idxs.detach().cpu().numpy()
        topk_probs = topk_probs.detach().cpu().numpy()

        new_path_pool, new_probs_pool = [], []
        for row in range(topk_idxs.shape[0]):
            path = path_pool[row]
            probs = probs_pool[row]
            for idx, p in zip(topk_idxs[row], topk_probs[row]):
                if idx >= len(acts_pool[row]):  # act idx is invalid
                    continue
                relation, next_node_id = acts_pool[row][idx]  # (relation, next_node_id)
                if relation == SELF_LOOP:
                    next_node_type = path[-1][1]
                else:
                    next_node_type = kg_relation[path[-1][1]][relation]
                new_path = path + [(relation, next_node_type, next_node_id)]
                new_path_pool.append(new_path)
                new_probs_pool.append(probs + [p])
        path_pool = new_path_pool
        probs_pool = new_probs_pool
        if hop < 2:
            state_pool = env._batch_get_state(path_pool)

    return path_pool, probs_pool


def predict_paths(policy_file, path_file, args):
    print('Predicting paths...')
    kg_relation = json.load(open(args.data_dir + "/kg_relation.json", 'r'))
    env = BatchKGEnvironment(args)
    pretrain_sd = torch.load(policy_file)
    model = ActorCritic(env.state_dim, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
    model_sd = model.state_dict()
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)

    test_labels = load_labels(args.data_dir, 'test')
    test_head_ids = list(test_labels.keys())

    batch_size = args.batch_size
    start_idx = 0
    all_paths, all_probs = [], []
    pbar = tqdm(total=len(test_head_ids))
    while start_idx < len(test_head_ids):
        end_idx = min(start_idx + batch_size, len(test_head_ids))
        batch_head_ids = test_head_ids[start_idx:end_idx]
        paths, probs = batch_beam_search(env, model, batch_head_ids, args.device, kg_relation, topk=args.topk)
        all_paths.extend(paths)
        all_probs.extend(probs)
        start_idx = end_idx
        pbar.update(batch_size)
    # predicts = pd.DataFrame({'paths': all_paths, 'probs': all_probs})
    # predicts.to_csv(path_file, index=False)
    predicts = {'paths': all_paths, 'probs': all_probs}
    pickle.dump(predicts, open(path_file, 'wb'))


def evaluate_paths(args, path_file, train_labels, test_labels):
    embeds = load_embed(args.data_dir)
    head_embeds = embeds[args.head_entity]
    relation_embeds = embeds[args.relation_type][0]
    tail_embeds = embeds[args.tail_entity]
    scores = np.dot(head_embeds + relation_embeds, tail_embeds.T)

    # 1) Get all valid paths for each user, compute path score and path probability.
    results = pickle.load(open(path_file, 'rb'))
    # results = pd.read_csv(path_file)
    pred_paths = {head_id: {} for head_id in test_labels}
    for path, probs in zip(results['paths'], results['probs']):
        if path[-1][1] != args.tail_entity:
            continue
        head_id = path[0][2]
        if head_id not in pred_paths:
            continue
        tail_id = path[-1][2]
        if tail_id not in pred_paths[head_id]:
            pred_paths[head_id][tail_id] = []
        path_score = scores[head_id][tail_id]
        path_prob = reduce(lambda x, y: x * y, probs)
        pred_paths[head_id][tail_id].append((path_score, path_prob, path))

    # 2) Pick best path for each head-tail pair, also remove tail_id if it is in train set.
    best_pred_paths = {}
    for head_id in pred_paths:
        train_tail_ids = set(train_labels[head_id])
        best_pred_paths[head_id] = []
        for tail_id in pred_paths[head_id]:
            if tail_id in train_tail_ids:
                continue
            # Get the path with highest probability
            sorted_path = sorted(pred_paths[head_id][tail_id], key=lambda x: x[1], reverse=True)
            best_pred_paths[head_id].append(sorted_path[0])

    # 3) Compute top 10 recommended products for each head_entity.
    pred_labels = {}
    result_sorted_list = []
    for head_id in best_pred_paths:
        if args.sort_by == 'score':
            sorted_path = sorted(best_pred_paths[head_id], key=lambda x: (x[0], x[1]), reverse=True)
        elif args.sort_by == 'prob':
            sorted_path = sorted(best_pred_paths[head_id], key=lambda x: (x[1], x[0]), reverse=True)
        predict_tail_ids = [p[-1][2] for _, _, p in sorted_path]# from largest to smallest
        top10_tail_ids = [p[-1][2] for _, _, p in sorted_path[:10]]  # from largest to smallest
        result_sorted_list.extend([[p, prob, score, tail] for (score, prob, p), tail in zip(sorted_path, predict_tail_ids)])
        # add up to 10 tail_ids if not enough
        if args.add_products and len(top10_tail_ids) < 10:
            train_tail_ids = set(train_labels[head_id])
            cand_tail_ids = np.argsort(scores[head_id])
            for cand_tail_id in cand_tail_ids[::-1]:
                if cand_tail_id in train_tail_ids or cand_tail_id in top10_tail_ids:
                    continue
                top10_tail_ids.append(cand_tail_id)
                if len(top10_tail_ids) >= 10:
                    break
        # end of add
        pred_labels[head_id] = predict_tail_ids  # change order to from smallest to largest!

    result_sorted_df = pd.DataFrame(result_sorted_list, columns=["path", "prob", 'embed_score', 'pred_tail'])
    result_sorted_df.to_csv(f"{args.output_dir}/path_sorted_df.csv", index=False)
    # evaluate(pred_labels, test_labels)
    metric_df = metrics_hit(pred_labels, test_labels, args.output_dir)


def test(args):
    policy_file = args.model_path
    path_file = args.log_dir + '/policy_paths_epoch.pkl'

    train_labels = load_labels(args.data_dir, mode='train')
    test_labels = load_labels(args.data_dir, mode='test')

    if args.run_path:
        predict_paths(policy_file, path_file, args)
    if args.run_eval:
        evaluate_paths(args, path_file, train_labels, test_labels)


if __name__ == '__main__':
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', type=str, default='/home/linjie/projects/KG/PGPR_hetionet/data',
    #                     help='data directory')
    # parser.add_argument('--dataset', type=str, default='hetionet', help='dataset name')
    # parser.add_argument('--output_dir', type=str, default='/home/linjie/projects/KG/PGPR_hetionet/output',
    #                     help='output directory')
    # parser.add_argument('--head_entity', type=str, default='Compound', help='head entity name to use')
    # parser.add_argument('--tail_entity', type=str, default='Compound', help='tail entity name to use')
    # parser.add_argument('--relation_type', type=str, default='CmC', help='relation type name to use')
    parser.add_argument('--model_path', type=str, required=True, help='model path')
    parser.add_argument('--sort_by', type=str, choices=['score', 'prob'], default='prob', help='predict path sort by')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=2, help='num of epochs.')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size.')
    parser.add_argument('--add_products', type=boolean, default=False, help='Add predicted products up to 10')
    parser.add_argument('--topk', type=int, nargs='*', default=[25, 5, 5, 1], help='number of samples')
    parser.add_argument('--run_path', type=boolean, default=True, help='Generate predicted path? (takes long time)')
    parser.add_argument('--run_eval', type=boolean, default=True, help='Run evaluation?')
    parser.add_argument("--config_file", type=str, help="train argparse config file",
                        default="output/args.json")
    args = parser.parse_args()
    # args.data_dir = args.data_path + '/' + args.dataset
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    for k, v in config.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else 'cpu'

    args.log_dir = args.output_dir
    test(args)

