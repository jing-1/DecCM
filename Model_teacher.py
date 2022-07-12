import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


EPS = 1e-15
MAX_LOGVAR = 10

class Encoder(nn.Module):
    def __init__(self, feature_dim, latent_dim, dropout_rate):
        super(Encoder, self).__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(feature_dim, latent_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc_mu = nn.Linear(feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(feature_dim, latent_dim)
        self.bn1 = nn.BatchNorm1d(feature_dim)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc_mu.weight)
        nn.init.xavier_normal_(self.fc_logvar.weight)



    def forward(self, features):
        features_ = self.dropout1(self.bn1(features))
        mu = self.fc_mu(features_)
        mu = F.normalize(mu, dim=1, p=2, eps=1e-10)
        logvar = self.fc_logvar(features_)
        logvar = F.normalize(logvar, dim=1, p=2, eps=1e-10)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, feature_dim, latent_dim, dropout_rate):
        super(Decoder, self).__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(latent_dim, latent_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(latent_dim)

        self.fc2 = nn.Linear(latent_dim, feature_dim)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)


    def forward(self, encoded_features):
        features_ = self.dropout1(self.bn1(encoded_features))
        features = F.relu(self.fc2(features_))
        return features


class t_net(nn.Module):
    def __init__(self, device, m_feature_dim, v_feature_dim, t_feature_dim, latent_dim, dropout_rate):
        super(t_net, self).__init__()
        self.device = device
        self.m_encoder = Encoder(m_feature_dim, latent_dim, dropout_rate)
        self.m_decoder = Decoder(m_feature_dim, latent_dim, dropout_rate)

        self.v_encoder = Encoder(v_feature_dim, latent_dim, dropout_rate)
        self.video_v_decoder = Decoder(v_feature_dim, latent_dim, dropout_rate)

        self.latent_dim = latent_dim
        self.mseloss = nn.MSELoss()

    def forward(self, batch_music, batch_video_v):

        self.m_mu, m_logvar = self.m_encoder(batch_music)
        v_mu, v_logvar = self.v_encoder(batch_video_v)

        m_z = self.reparametrize(self.m_mu, m_logvar)

        v_z = self.reparametrize(v_mu, v_logvar)


        kl_loss_m = self.kl_loss(self.m_mu, m_logvar)
        kl_loss_v = self.kl_loss(v_mu, v_logvar)

        m_v_features = self.m_decoder(v_z)
        video_v_features = self.video_v_decoder(m_z)
        return kl_loss_m, kl_loss_v, m_v_features, video_v_features, m_z,  v_z, self.m_mu, m_logvar, v_mu, v_logvar

    def reparametrize(self, mu, logvar, sigmoid=True):
        logvar = logvar.clamp(max=MAX_LOGVAR)
        if self.training:
            return mu + torch.randn_like(logvar) * torch.exp(logvar.mul(0.5))
        else:
            return mu

    def recon_loss(self, videos_v, musics, video_recon_m_features, m_recon_v_features):
        scores = self.mseloss(video_recon_m_features, musics) + self.mseloss(m_recon_v_features, videos_v)

        return scores

    def matching_loss(self, m_z, v_z):
        scores = self.dot_product_decode(m_z, v_z, sigmoid=False)
        return scores


    def kl_loss(self, mu, logvar):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logvar`, or based on latent variables from last encoding.

        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logvar (Tensor, optional): The latent space for
                :math:`\log\sigma^2`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        logvar = logvar.clamp(max=MAX_LOGVAR)
        return -0.5 * torch.mean(
            torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1))

    def dot_product_decode(self, features, recon_features, sigmoid=True):
        value = torch.sum(torch.mul(features, recon_features), dim=-1)
        return torch.sigmoid(value) if sigmoid else value


    def pdist(self, x1, x2):
        """
                x1: Tensor of shape (h1, w)
                x2: Tensor of shape (h2, w)
                Return pairwise distance for each row vector in x1, x2 as
                a Tensor of shape (h1, h2)
            """

        return torch.mm(x1, torch.t(x2))


    def embedding_loss(self, data, args):
        videos_v, musics = data
        videos_v = torch.from_numpy(videos_v).float().to(self.device)
        musics = torch.from_numpy(musics).float().to(self.device)
        kl_loss_m, kl_loss_v, m_v_features, video_v_features, m_z,  v_z, _ , m_var, _, v_var= self.forward(musics, videos_v)
        num_music = musics.shape[0]
        num_video = videos_v.shape[0]
        aff_mv = np.eye(num_music, dtype=np.bool)

        mv_dist = self.pdist(m_z, v_z)
        #music loss
        pos_pair_dist = mv_dist[aff_mv].view([num_video, 1])
        neg_pair_dist = mv_dist[~aff_mv].view([num_video, -1])
        music_loss = torch.clamp(-args.margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
        music_loss = torch.mean(torch.topk(music_loss, k=args.num_neg_sample)[0])
        #video loss
        pos_pair_dist = torch.t(mv_dist)[aff_mv].view([num_video, 1])
        neg_pair_dist = torch.t(mv_dist)[~aff_mv].view([num_video, -1])
        video_loss = torch.clamp(-args.margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
        video_loss = torch.mean(torch.topk(video_loss, k=args.num_neg_sample)[0])

        loss = -music_loss * args.music_loss_factor - video_loss + kl_loss_m +  kl_loss_v + args.weight_recon*self.recon_loss(videos_v, musics, m_v_features, video_v_features)
        return loss, self.recon_loss(videos_v, musics, m_v_features, video_v_features), - video_loss


