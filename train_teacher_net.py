import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from DataLoad_teacher import MyDataset
from Model_teacher import t_net
import torch.nn.functional as F

import random

import logging.handlers
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger("logger")

handler1 = logging.StreamHandler()
handler2 = logging.FileHandler(filename="test.log")

logger.setLevel(logging.DEBUG)
handler1.setLevel(logging.DEBUG)
handler2.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
handler1.setFormatter(formatter)
handler2.setFormatter(formatter)

logger.addHandler(handler1)
logger.addHandler(handler2)


class Net:
    def __init__(self, args):
        ##########################################################################################################################################
        seed = args.seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        ##########################################################################################################################################
        self.model_name = args.model_name
        self.data_path = args.data_path
        self.learning_rate = args.l_r  # l_r#
        self.weight_decay = args.weight_decay  # weight_decay#
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_epoch = args.num_epoch

        self.dim_latent = args.dim_latent
        self.dim_v = 2048
        self.dim_m = 128 + 128
        self.dim_t = 768
        self.num_music = args.num_music
        self.topk = args.topk
        self.kl_start = args.kl_start
        self.dropout_rate = args.dropout_rate
        self.lr_update = args.lr_update

        ##########################################################################################################################################
        print('Data loading ...')
        self.train_dataset = MyDataset(self.data_path, self.num_music)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_workers)
        self.val_dataset = np.load(self.data_path + 'val.npy')
        self.v_feat = np.load(self.data_path + 'mv_video_resnet_features.npy')
        self.m_feat = np.load(self.data_path + 'vggish_features.npy')
        m_smile_feat = np.load(self.data_path + 'opensmile_features_norm.npy')
        self.m_feat = np.concatenate([self.m_feat, m_smile_feat], axis=1)

        print('Data has been loaded.')
        ##########################################################################################################################################

        if self.model_name == 'MM_CrossAE':
            self.model = t_net(self.device, self.dim_m, self.dim_v, self.dim_t, self.dim_latent, self.dropout_rate).to(
                self.device)

        if args.PATH_weight_load and os.path.exists(args.PATH_weight_load):
            self.model.load_state_dict(torch.load(args.PATH_weight_load))
            print('module weights loaded....')
        ##########################################################################################################################################
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': self.learning_rate}],
                                          weight_decay=self.weight_decay)
        ##########################################################################################################################################

    def run(self):

        max_val_epoch = 0
        max_val_recall = 0.
        kl_weight = self.kl_start
        anneal_rate = (1.0 - args.kl_start) / ((len(self.train_dataset) / self.batch_size))
        matching_losses = []
        recon_losses = []
        val_recalles = []
        for epoch in range(self.num_epoch):
            self.model.train()
            print('Now, training start ...')
            pbar = tqdm(total=len(self.train_dataset))

            sum_loss = 0.0
            recon_loss = 0.
            matching_loss = 0.
            for data in self.train_dataloader:

                videos, pos_musics, neg_musics = data
                kl_weight = min(1.0, kl_weight + anneal_rate)
                videos_v_f = self.v_feat[videos]
                pos_musics_f = self.m_feat[pos_musics]
                data = [videos_v_f, pos_musics_f]
                self.optimizer.zero_grad()
                self.loss, self.recon_loss, self.matching_loss = self.model.embedding_loss(data, args)
                self.loss.backward()
                self.optimizer.step()
                pbar.update(self.batch_size)
                sum_loss += self.loss.item()
                recon_loss += self.recon_loss.item()
                matching_loss += self.matching_loss.item()
            print(recon_loss)
            print(matching_loss)
            recon_losses.append(round(recon_loss, 4))
            matching_losses.append(round(matching_loss, 4))


            logger.info('loss:{}'.format(sum_loss / self.batch_size))

            pbar.close()

            print('Validation start...')
            self.model.eval()
            with torch.no_grad():
                recall = self.accuracy(self.val_dataset, topk=25)
                print(
                    '---------------------------------Val: {0}-th epoch {1}-th top  Recall:{2:.4f}---------------------------------'.format(
                        epoch, self.topk, recall))
                logger.info(
                    '---------------------------------Val: {0}-th epoch {1}-th top  Recall:{2:.4f}---------------------------------'.format(
                        epoch, self.topk, recall))
                if recall > max_val_recall:
                    max_val_recall = recall
                    max_val_epoch = epoch
                    torch.save(self.model.state_dict(), 'model/pgc/net_params_128.pkl')
                val_recalles.append(round(recall, 4))

        return max_val_epoch, max_val_recall

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR
           decayed by 10 every 30 epochs"""
        lr = self.learning_rate * (0.1 ** (epoch // self.lr_update))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def accuracy(self, dataset, topk=50):
        sum_item = len(dataset)

        all_m_mu, all_m_logvar = self.model.m_encoder(torch.tensor(self.m_feat).to(self.device))
        video = dataset[:, 0]
        pos_item = dataset[:, 1]
        video_v = self.v_feat[video]
        # video_t = self.t_feat[video]
        batch_video_v_tensor = torch.tensor(video_v).to(self.device)
        # batch_video_t_tensor = torch.tensor(video_t).to(self.device)
        v_mu, v_logvar= self.model.v_encoder(batch_video_v_tensor)
        # pd_mu = F.normalize(pd_mu, dim=1, p=2, eps=1e-10)
        # all_m_mu = F.normalize(all_m_mu, dim=1, p=2, eps=1e-10)

        video_music_sims = v_mu.mm(all_m_mu.t())
        _, index_of_rank_list = torch.topk(video_music_sims, topk, dim=1)
        index_of_rank_list = index_of_rank_list.cpu().numpy()
        # np.savetxt('index_of_rank_list.txt', index_of_rank_list)
        num_hit = [np.isin(pos_item[i], index_of_rank_list[i]) for i in range(len(dataset))]
        num_hit = np.array(num_hit).sum()/ sum_item

        return num_hit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=1234, help='for reproduce')
    parser.add_argument('--model_name', default='MM_CrossAE', help='Model name.')
    parser.add_argument('--data_path', default='data_pgc/', help='Dataset path')
    parser.add_argument('--PATH_weight_load', default='', help='Loading weight filename.')
    parser.add_argument('--PATH_weight_save', default='', help='Writing weight filename.')
    parser.add_argument('--l_r', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--dim_latent', type=int, default=128, help='Latent dimension.')
    parser.add_argument('--num_epoch', type=int, default=50, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=40, help='Workers number.')
    parser.add_argument('--num_music', type=int, default=3003, help='Number of all train musics')
    parser.add_argument('--topk', type=int, default=200)
    parser.add_argument('--kl_start', type=float, default=0.)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--num_neg_sample', type=int, default=40, help='Number of negative example to sample.')
    parser.add_argument('--margin', type=float, default=0.05, help='Margin.')
    parser.add_argument('--music_loss_factor', type=float, default=1.,
                        help='Factor multiplied with image loss. Set to 0 for single direction.')
    parser.add_argument('--lr_update', default=5, type=int,
                        help='Number of epochs to update the learning rate.')

    args = parser.parse_args()
    args.weight_recon = 1

    egcn = Net(args)
    max_val_epoch, max_val_recall = egcn.run()
