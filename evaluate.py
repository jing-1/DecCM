import numpy as np
import torch
from Model import s_net

import math
import logging
logger = logging.getLogger("logger")



class Evaluate():
    def __init__(self):
        super(Evaluate, self).__init__()
        self.data_path = 'data_matching/'
        self.v_feat = np.load(self.data_path + 'visual_features_resnet_V.npy')
        self.m_feat = np.load(self.data_path + 'music_features.npy')
        m_smile_feat = np.load(self.data_path + 'new_music_smile_features_norm.npy')
        self.m_feat = np.concatenate([self.m_feat, m_smile_feat], axis=1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.confounder_prior = np.load(self.data_path  + 'user_prior_mean.npy')


        self.dim_m =128 + 128
        self.dim_v = 2048
        self.dim_t = 100
        self.dim_latent = 128
        self.dropout_rate = 0.2
        self.num_music = 3003
        self.batch_size = 1024
        self.num_workers = 40
        self.num_groups = 6
        self.model = s_net(self.device, self.dim_m, self.dim_v, self.dim_t, self.dim_latent, self.num_groups, self.dropout_rate, self.confounder_prior).to(self.device)
        self.model.load_state_dict(torch.load('model/net_params.pkl'))

        self.test_dataset = np.load(self.data_path + '/100neg/' + 'test.npy', allow_pickle=True)
        self.weight = np.load(self.data_path + 'video_popularity_test.npy')


    def forward(self):
        self.model.eval()
        with torch.no_grad():
            self.model.eval()
            with torch.no_grad():
                for idx, topk in enumerate([10, 15, 20, 25]):
                    recall, ndcg = self.accuracy_neg(self.test_dataset, self.weight, topk=topk)
                    print(
                        '---------------------------------Test: {0}-th top  Recall:{1:.4f}  Ndcg:{2:.4f}---------------------------------'.format(
                            topk, recall, ndcg))

    def accuracy_neg(self, dataset, popularity, topk=50, num_neg=100):
        sum_item = len(dataset)

        all_m_mu, all_m_logvar = self.model.m_encoder(torch.tensor(self.m_feat).to(self.device))
        video = dataset[:, 0].astype(int)
        pos_item = dataset[:, 1].astype(int)
        neg_items = np.array([dataset[i][2].tolist() for i in range(sum_item)])
        items = np.hstack([np.expand_dims(pos_item, axis=1), neg_items])
        video_v = self.v_feat[video]
        batch_video_v_tensor = torch.tensor(video_v).to(self.device)
        v_mu, v_logvar = self.model.v_encoder(batch_video_v_tensor)

        confounder_prior = torch.tensor(self.confounder_prior, dtype=torch.float32).to(self.device).unsqueeze(dim=-1)
        weighted_confounder_emb = torch.sum(self.model.confounder_emb(
            torch.tensor(np.arange(self.num_groups)).to(self.device).repeat(len(all_m_mu),
                                                                            1)) * confounder_prior, dim=1)
        all_m_mu = self.model.linear(torch.cat([weighted_confounder_emb, all_m_mu], -1))

        video_music_sims = v_mu.mm(all_m_mu.t())

        rows = np.arange(sum_item)
        rows = np.reshape(rows.repeat(num_neg + 1), [sum_item, num_neg + 1])
        video_music_sims = video_music_sims[rows, items]

        ranked_music_ids, index_of_rank_list = torch.topk(video_music_sims, topk, dim=1)
        index_of_rank_list = index_of_rank_list.cpu().numpy()

        num_hit = popularity[[np.isin(0, index_of_rank_list[i]) for i in range(len(dataset))]]
        num_hit = np.array(num_hit).sum()

        ###cal ndcg
        sum_ndcg = 0.0
        for idx, index_ in enumerate(index_of_rank_list):
            if 0 in index_:
                index = list(index_).index(0)
                ndcg_score = math.log(2) / math.log(index + 2)
                sum_ndcg += popularity[idx] * ndcg_score

        return num_hit, sum_ndcg


evaluate = Evaluate()
evaluate.forward()
