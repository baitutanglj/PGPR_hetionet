import random
import json
import numpy as np
from utils import load_embed, load_kg, metapaths


class KGState(object):
    def __init__(self, embed_size, history_len=1):
        self.embed_size = embed_size
        self.history_len = history_len  # mode: one of {full, current}
        if history_len == 0:
            self.dim = 2 * embed_size
        elif history_len == 1:
            self.dim = 4 * embed_size
        elif history_len == 2:
            self.dim = 6 * embed_size
        else:
            raise Exception('history length should be one of {0, 1, 2}')

    def __call__(self, head_entity_embed, node_embed, last_node_embed, last_relation_embed, older_node_embed,
                 older_relation_embed):
        if self.history_len == 0:
            return np.concatenate([head_entity_embed, node_embed])
        elif self.history_len == 1:
            return np.concatenate([head_entity_embed, node_embed, last_node_embed, last_relation_embed])
        elif self.history_len == 2:
            return np.concatenate([head_entity_embed, node_embed, last_node_embed, last_relation_embed, older_node_embed,
                                   older_relation_embed])
        else:
            raise Exception('mode should be one of {full, current}')


class BatchKGEnvironment(object):
    def __init__(self, args):
        self.max_acts = args.max_acts
        self.act_dim = args.max_acts + 1  # Add self-loop action, whose act_idx is always 0.
        self.max_num_nodes = args.max_path_len + 1 # max number of hops (= #nodes - 1)
        self.state_history = args.state_history
        self.kg = load_kg(args.data_dir)
        self.self_loop = 'self_loop'
        self.embeds = load_embed(args.data_dir)
        self.head_entity = args.head_entity
        self.tail_entity = args.tail_entity
        self.relation_type = args.relation_type
        self.kg_relation = json.load(open(args.data_dir + "/kg_relation.json", 'r'))
        
        self.embed_size = self.embeds[args.head_entity].shape[1]
        self.embeds[self.self_loop] = (np.zeros(self.embed_size), 0.0)
        self.state_gen = KGState(self.embed_size, history_len=self.state_history)
        self.state_dim = self.state_gen.dim

        # Compute head_entity-tail_entity scores for scaling.
        u_p_scores = np.dot(self.embeds[self.head_entity] + self.embeds[self.relation_type][0], self.embeds[self.tail_entity].T)
        self.u_p_scales = np.max(u_p_scores, axis=1)

        # Compute path patterns
        self.metapaths = []
        for metapath_id, metapath in metapaths.items():
            metapath = [self.self_loop] + [v[0] for v in metapath[1:]]  # pattern contains all relations
            for i in range(args.max_path_len+1-len(metapath)):
                metapath.append(self.self_loop)
            self.metapaths.append(tuple(metapath))

        # Following is current episode information.
        self._batch_path = None  # list of tuples of (relation, node_type, node_id)
        self._batch_curr_actions = None  # save current valid actions
        self._batch_curr_state = None
        self._batch_curr_reward = None
        # Here only use 1 'done' indicator, since all paths have same length and will finish at the same time.
        self._done = False

    def _has_pattern(self, path):
        pattern = tuple([v[0] for v in path])
        return pattern in self.metapaths

    def _batch_has_pattern(self, batch_path):
        return [self._has_pattern(path) for path in batch_path]

    def _get_actions(self, path, done):
        """Compute actions for current node."""
        _, curr_node_type, curr_node_id = path[-1]
        actions = [(self.self_loop, curr_node_id)]  # self-loop must be included.

        # (1) If game is finished, only return self-loop action.
        if done:
            return actions

        # (2) Get all possible edges from original knowledge graph.
        # [CAVEAT] Must remove visited nodes!
        relations_nodes = self.kg(curr_node_type, curr_node_id)
        candidate_acts = []  # list of tuples of (relation, node_type, node_id)
        visited_nodes = set([(v[1], v[2]) for v in path])
        for r, next_node_ids in relations_nodes.items():
            if len(next_node_ids)>0:
                next_node_type = self.kg_relation[curr_node_type][r]
                next_node_ids = [n for n in next_node_ids if (next_node_type, n) not in visited_nodes]  # filter
                candidate_acts.extend(zip([r] * len(next_node_ids), next_node_ids))

        # (3) If candidate action set is empty, only return self-loop action.
        if len(candidate_acts) == 0:
            return actions

        # (4) If number of available actions is smaller than max_acts, return action sets.
        if len(candidate_acts) <= self.max_acts:
            candidate_acts = sorted(candidate_acts, key=lambda x: (x[0], x[1]))
            actions.extend(candidate_acts)
            return actions

        # (5) If there are too many actions, do some deterministic trimming here!
        head_entity_embed = self.embeds[self.head_entity][path[0][-1]]
        candidate_acts_ = np.array(candidate_acts)
        rel_type_mask = candidate_acts_[:,0]==self.relation_type

        scores = []
        for r, next_node_id in candidate_acts:
            next_node_type = self.kg_relation[curr_node_type][r]
            if next_node_type == self.head_entity:
                src_embed = head_entity_embed
            elif next_node_type == self.tail_entity:
                src_embed = head_entity_embed + self.embeds[self.relation_type][0]
            else:
                src_embed = head_entity_embed + self.embeds[self.relation_type][0] + self.embeds[r][0]
            score = np.matmul(src_embed, self.embeds[next_node_type][next_node_id])#矩阵乘法
            if r =='CcSE':
                score = 0.5 * score
            # This trimming may filter out target tail_entitys!
            # Manually set the score of target tail_entitys a very large number.
            # if next_node_type == tail_entity and next_node_id in self._target_pids:
            #    score = 99999.0
            scores.append(score)
        candidate_idxs = np.argsort(scores)[-self.max_acts:]  # choose actions with larger scores ##np.argsort从小到大排序，[-self.max_acts:]：取得分大的action
        candidate_acts = sorted([candidate_acts[i] for i in candidate_idxs], key=lambda x: (x[0], x[1]))
        actions.extend(candidate_acts)
        return actions

    def _batch_get_actions(self, batch_path, done):
        return [self._get_actions(path, done) for path in batch_path]

    def _get_state(self, path):
        """Return state of numpy vector: [head_entity_embed, curr_node_embed, last_node_embed, last_relation]."""
        head_entity_embed = self.embeds[self.head_entity][path[0][-1]]
        zero_embed = np.zeros(self.embed_size)
        if len(path) == 1:  # initial state
            state = self.state_gen(head_entity_embed, head_entity_embed, zero_embed, zero_embed, zero_embed, zero_embed)
            return state

        older_relation, last_node_type, last_node_id = path[-2]
        last_relation, curr_node_type, curr_node_id = path[-1]
        curr_node_embed = self.embeds[curr_node_type][curr_node_id]
        last_node_embed = self.embeds[last_node_type][last_node_id]
        last_relation_embed = self.embeds[last_relation][0]  # this can be self-loop!
        if len(path) == 2:
            state = self.state_gen(head_entity_embed, curr_node_embed, last_node_embed, last_relation_embed, zero_embed,
                                   zero_embed)
            return state

        _, older_node_type, older_node_id = path[-3]
        older_node_embed = self.embeds[older_node_type][older_node_id]
        older_relation_embed = self.embeds[older_relation][0]
        state = self.state_gen(head_entity_embed, curr_node_embed, last_node_embed, last_relation_embed, older_node_embed,
                               older_relation_embed)
        return state

    def _batch_get_state(self, batch_path):
        batch_state = [self._get_state(path) for path in batch_path]
        return np.vstack(batch_state)  # [bs, dim]

    def _get_reward(self, path):
        # If it is initial state or 1-hop search, reward is 0.
        if len(path) <= 2:
            return 0.0

        if not self._has_pattern(path):
            return 0.0

        target_score = 0.0
        _, curr_node_type, curr_node_id = path[-1]
        if curr_node_type == self.tail_entity:
            # Give soft reward for other reached tail_entitys.
            head_entity_id = path[0][-1]
            u_vec = self.embeds[self.head_entity][head_entity_id] + self.embeds[self.relation_type][0]
            p_vec = self.embeds[self.tail_entity][curr_node_id]
            score = np.dot(u_vec, p_vec) / self.u_p_scales[head_entity_id]
            target_score = max(score, 0.0)

        return target_score

    def _batch_get_reward(self, batch_path):
        batch_reward = [self._get_reward(path) for path in batch_path]
        return np.array(batch_reward)

    def _is_done(self):
        """Episode ends only if max path length is reached."""
        return self._done or len(self._batch_path[0]) >= self.max_num_nodes

    def reset(self, head_entity_ids=None):
        if head_entity_ids is None:
            all_head_entity_ids = list(self.kg(self.head_entity).keys())
            head_entity_ids = [random.choice(all_head_entity_ids)]

        # each element is a tuple of (relation, entity_type, entity_id)
        self._batch_path = [[(self.self_loop, self.head_entity, head_entity_id)] for head_entity_id in head_entity_ids]
        self._done = False
        self._batch_curr_state = self._batch_get_state(self._batch_path)
        self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done)
        self._batch_curr_reward = self._batch_get_reward(self._batch_path)

        return self._batch_curr_state

    def batch_step(self, batch_act_idx):
        """
        Args:
            batch_act_idx: list of integers.
        Returns:
            batch_next_state: numpy array of size [bs, state_dim].
            batch_reward: numpy array of size [bs].
            done: True/False
        """
        assert len(batch_act_idx) == len(self._batch_path)

        # Execute batch actions.
        for i in range(len(batch_act_idx)):
            act_idx = batch_act_idx[i]
            _, curr_node_type, curr_node_id = self._batch_path[i][-1]
            relation, next_node_id = self._batch_curr_actions[i][act_idx]
            if relation == self.self_loop:
                next_node_type = curr_node_type
            else:
                next_node_type = self.kg_relation[curr_node_type][relation]
            self._batch_path[i].append((relation, next_node_type, next_node_id))

        self._done = self._is_done()  # must run before get actions, etc.
        self._batch_curr_state = self._batch_get_state(self._batch_path)
        self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done)
        self._batch_curr_reward = self._batch_get_reward(self._batch_path)

        return self._batch_curr_state, self._batch_curr_reward, self._done

    def batch_action_mask(self, dropout=0.0):
        """Return action masks of size [bs, act_dim]."""
        batch_mask = []
        for actions in self._batch_curr_actions:
            act_idxs = list(range(len(actions)))
            if dropout > 0 and len(act_idxs) >= 5:
                keep_size = int(len(act_idxs[1:]) * (1.0 - dropout))
                tmp = np.random.choice(act_idxs[1:], keep_size, replace=False).tolist()
                act_idxs = [act_idxs[0]] + tmp
            act_mask = np.zeros(self.act_dim, dtype=np.uint8)
            act_mask[act_idxs] = 1
            batch_mask.append(act_mask)
        return np.vstack(batch_mask)

    def print_path(self):
        for path in self._batch_path:
            msg = 'Path: {}({})'.format(path[0][1], path[0][2])
            for node in path[1:]:
                msg += ' =={}=> {}({})'.format(node[0], node[1], node[2])
            print(msg)
