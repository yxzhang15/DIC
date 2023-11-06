import os
import sys
import glob
import h5py
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from util import knn, batch_choice
from crossatt import AttentionalPropagation
from scipy.spatial.transform import Rotation
#from vis import get_colored_point_cloud_feature
#import matplotlib.pyplot as plt


class Conv1DBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, ksize):
        super(Conv1DBNReLU, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, ksize, bias=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1DBlock(nn.Module):
    def __init__(self, channels, ksize):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.conv.append(Conv1DBNReLU(channels[i], channels[i + 1], ksize))
        self.conv.append(nn.Conv1d(channels[-2], channels[-1], ksize))

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x


class Conv2DBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, ksize):
        super(Conv2DBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, ksize, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv2DBlock(nn.Module):
    def __init__(self, channels, ksize):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.conv.append(Conv2DBNReLU(channels[i], channels[i + 1], ksize))
        self.conv.append(nn.Conv2d(channels[-2], channels[-1], ksize))

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x


class Propagate(nn.Module):
    def __init__(self, in_channel, emb_dims):
        super(Propagate, self).__init__()
        self.conv2d = Conv2DBlock((in_channel, emb_dims, emb_dims), 1)
        self.conv1d = Conv1DBlock((emb_dims, emb_dims), 1)

    def forward(self, x, idx):
        batch_idx = np.arange(x.size(0)).reshape(x.size(0), 1, 1)
        nn_feat = x[batch_idx, :, idx].permute(0, 3, 1, 2)
        x = nn_feat - x.unsqueeze(-1)
        x = self.conv2d(x)
        x = x.max(-1)[0]
        x = self.conv1d(x)
        return x


class TemperatureNet1(nn.Module):
    def __init__(self):
        super(TemperatureNet1, self).__init__()
        self.nn = nn.Sequential(nn.Linear(3, 3),
                                nn.BatchNorm1d(3),
                                nn.ReLU(),
                                nn.Linear(3, 3),
                                nn.BatchNorm1d(3),
                                nn.ReLU(),
                                nn.Linear(3, 1),
                                nn.ReLU())

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src = src.mean(dim=2)
        tgt = tgt.mean(dim=2)
        residual = torch.abs(src - tgt)

        return torch.clamp(self.nn(residual), 1.0 / 20, 1.0 * 20)


class TemperatureNet_high1(nn.Module):
    def __init__(self):
        super(TemperatureNet_high1, self).__init__()
        self.nn = nn.Sequential(nn.Linear(64, 64),
                                nn.BatchNorm1d(64),
                                nn.ReLU(),
                                nn.Linear(64, 16),
                                nn.BatchNorm1d(16),
                                nn.ReLU(),
                                nn.Linear(16, 1),
                                nn.ReLU())

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src = src.mean(dim=2)
        tgt = tgt.mean(dim=2)
        residual = torch.abs(src - tgt)

        return torch.clamp(self.nn(residual), 1.0 / 20, 1.0 * 20)


class TemperatureNet_high(nn.Module):
    def __init__(self, num_preserved_point):
        super(TemperatureNet_high, self).__init__()
        self.nn = nn.Sequential(nn.Linear(num_preserved_point, num_preserved_point),
                                nn.BatchNorm1d(num_preserved_point),
                                nn.ReLU(),
                                nn.Linear(num_preserved_point, num_preserved_point),
                                nn.BatchNorm1d(num_preserved_point),
                                nn.ReLU(),
                                nn.Linear(num_preserved_point, num_preserved_point),
                                nn.ReLU())

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src = src.mean(dim=1)
        tgt = tgt.mean(dim=1)
        residual = torch.abs(src - tgt)

        residual = self.nn(residual)
        return torch.clamp(residual, 1.0 / 20, 1.0 * 20)


class position_encode_origin(nn.Module):
    def __init__(self):
        super(position_encode_origin, self).__init__()

        self.pe1 = Conv1DBlock((64, 64, 64), 1)

    def forward(self, src_embedding, tgt_embedding, tau_high):
        batch_size, num_dims, num_points = x.size()
        similarity_matrix_coarse = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(
            num_dims)

        similarity_matrix_coarse = similarity_matrix_coarse.view(batch_size * num_points, num_points)
        similarity_matrix_coarse = F.gumbel_softmax(similarity_matrix_coarse, tau_high, hard=True)
        similarity_matrix_coarse = similarity_matrix_coarse.view(batch_size, num_points, num_points)

        max_src_idx, max_tgt_idx = find_most_similar(similarity_matrix_coarse)
        src_point = src_embedding[batch_idx, :, max_src_idx].transpose(1, 2)
        tgt_point = tgt_embedding[batch_idx, :, max_tgt_idx].transpose(1, 2)

        src_embedding_dist = src_embedding - src_point.repeat(1, 1, num_points)
        tgt_embedding_dist = tgt_embedding - tgt_point.repeat(1, 1, num_points)
        src_embedding_pe = self.pe1(src_embedding_dist)
        tgt_embedding_pe = self.pe1(tgt_embedding_dist)
        src_embedding = src_embedding + src_embedding_pe
        tgt_embedding = tgt_embedding + tgt_embedding_pe
        return src_embedding, tgt_embedding, src_embedding_dist, tgt_embedding_dist


class GNN(nn.Module):
    def __init__(self, emb_dims=64):
        super(GNN, self).__init__()
        self.propogate1 = Propagate(3, 64)
        self.propogate2 = Propagate(64, 64)
        self.propogate3 = Propagate(64, 64)
        self.propogate4 = Propagate(64, 64)
        self.propogate5 = Propagate(64, emb_dims)
        self.att1 = AttentionalPropagation(64, 4)
        self.att2 = AttentionalPropagation(64, 4)
        self.att3 = AttentionalPropagation(64, 4)
        self.conv = Conv1DBlock((emb_dims * 2, emb_dims), 1)

    def forward(self, x):
        nn_idx = knn(x, k=12)
        x1 = self.propogate1(x, nn_idx)
        x2 = self.propogate2(x1, nn_idx)
        x2 = x1 + x2
        x3 = self.propogate3(x2, nn_idx)
        x3 = x1 + x2 + x3
        x1_embedding = self.att1(x1, x1, False)
        x2_embedding = self.att2(x2, x2, False)
        x3_embedding = self.att3(x3, x3, False)
        embedding = x1_embedding + x2_embedding + x3_embedding
        embedding = torch.softmax(embedding, dim=2)
        embedding = torch.exp(embedding)
        embedding = embedding * x3
        x = x3 + embedding
        return x


class GNN1(nn.Module):
    def __init__(self, emb_dims=64):
        super(GNN1, self).__init__()
        self.propogate1 = Propagate(3, 64)
        self.propogate2 = Propagate(64, 64)
        self.propogate3 = Propagate(64, 64)
        self.propogate4 = Propagate(64, 64)
        self.propogate5 = Propagate(64, emb_dims)
        self.att1 = AttentionalPropagation(64, 4)
        self.att2 = AttentionalPropagation(64, 4)
        self.att3 = AttentionalPropagation(64, 4)
        self.conv = Conv1DBlock((emb_dims * 2, emb_dims), 1)

    def forward(self, x):
        nn_idx = knn(x, k=12)
        x1 = self.propogate1(x, nn_idx)
        x2 = self.propogate2(x1, nn_idx)
        x3 = self.propogate3(x2, nn_idx)
        return x3


class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.emb_dims = args.emb_dims
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, src, tgt, weights):

        var_eps = 1e-3

        weights = weights.unsqueeze(2)  # 16 128 1
        src = src.transpose(1, 2)  # b 768 3
        srcm = torch.matmul(weights.transpose(1, 2), src)  # b 1 3
        src_centered = src - srcm  # b 128 3
        src_centered = src_centered.transpose(1, 2)

        tgt = tgt.transpose(1, 2)  # b 768 3
        tgtm = torch.matmul(weights.transpose(1, 2), tgt)  # b 1 3
        tgt_centered = tgt - tgtm  # b 128 3
        tgt_centered = tgt_centered.transpose(1, 2)

        weight_matrix = torch.diag_embed(weights.squeeze(2))

        H = torch.matmul(src_centered, torch.matmul(weight_matrix, tgt_centered.transpose(2, 1).contiguous()))

        U, S, V = [], [], []
        R = []

        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
            R.append(r)

            U.append(u)
            S.append(s)
            V.append(v)

        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, srcm.transpose(1, 2)) + tgtm.transpose(1, 2)
        return R, t.view(src.size(0), 3)
        return R, t.view(batch_size, 3)


def count_ones(matrix):
    count = 0
    for row in matrix:
        for element in row:
            if element == 1:
                count += 1
    return count


def find_most_similar(matrix_batch):
    batch_size, rows, cols = matrix_batch.size()

    most_similar_rows = []
    most_similar_cols = []

    for i in range(batch_size):
        matrix = matrix_batch[i]
        max_similarity_index = torch.argmax(matrix)
        most_similar_row = max_similarity_index // cols
        most_similar_col = max_similarity_index % cols

        most_similar_rows.append(most_similar_row.item())
        most_similar_cols.append(most_similar_col.item())

    most_similar_rows = [[x] for x in most_similar_rows]
    most_similar_cols = [[x] for x in most_similar_cols]
    return most_similar_rows, most_similar_cols


def square_dists(points1, points2):
    '''
    Calculate square dists between two group points
    :param points1: shape=(B, N, C)
    :param points2: shape=(B, M, C)
    :return:
    '''
    B, N, C = points1.shape
    _, M, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, N, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, M)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    # dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists) # Very Important for dist = 0.
    return dists.float()


def dists_for_sim(x, y):
    diff_origin = x.unsqueeze(-1) - y.unsqueeze(-2)
    dist_origin = (diff_origin ** 2).sum(1, keepdim=True)
    dist_origin = torch.sqrt(dist_origin)
    diff_origin = diff_origin / (dist_origin + 1e-8)
    return dist_origin, diff_origin


def Ol_loss(x_ol, y_ol, dists):
    CELoss = nn.CrossEntropyLoss()
    # CELoss = F.binary_cross_entropy_with_logit()
    dists = dists.sum(1)
    x_ol_gt = (torch.min(dists, dim=-1)[0] < 0.05 * 0.05).long()  # (B, N)
    y_ol_gt = (torch.min(dists, dim=1)[0] < 0.05 * 0.05).long()  # (B, M)
    x_ol_loss = CELoss(x_ol, x_ol_gt)
    y_ol_loss = CELoss(y_ol, y_ol_gt)
    ol_loss = (x_ol_loss + y_ol_loss) / 2
    return ol_loss


def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_matrix(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')


def rotation_matrix_to_euler(rotation_matrix):
    # 计算旋转矩阵的欧拉角表示
   # batch_size, _ = rotation_matrix.size()
    euler_angles = torch.zeros(1, 3)
    
    
    rotation_matrix = torch.tensor(rotation_matrix)
    # 提取旋转矩阵的元素
    r00 = rotation_matrix[:, 0, 0]
    r01 = rotation_matrix[:, 0, 1]
    r02 = rotation_matrix[:, 0, 2]
    r10 = rotation_matrix[:, 1, 0]
    r11 = rotation_matrix[:, 1, 1]
    r12 = rotation_matrix[:, 1, 2]
    r20 = rotation_matrix[:, 2, 0]
    r21 = rotation_matrix[:, 2, 1]
    r22 = rotation_matrix[:, 2, 2]

    # 计算欧拉角表示
    euler_angles[:, 0] = torch.atan2(r21, r22)  # 绕X轴的旋转
    euler_angles[:, 1] = torch.atan2(-r20, torch.sqrt(r21**2 + r22**2))  # 绕Y轴的旋转
    euler_angles[:, 2] = torch.atan2(r10, r00)  # 绕Z轴的旋转

    return euler_angles

def tsne(point_cloud_features):
    point_cloud_features = point_cloud_features[0]
    tsne = TSNE(n_components=2, random_state=42)
    point_cloud_features = point_cloud_features.cuda().cpu()
    # 对点云特征进行降维
    reduced_features = tsne.fit_transform(point_cloud_features)
    
    # 绘制降维后的点云
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1])
    plt.title('t-SNE Visualization of Point Cloud Features')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig('./tsne', dpi=600)


class DNET(nn.Module):
    def __init__(self, args):
        super(DNET, self).__init__()
        self.emb_dims = args.emb_dims
        self.num_iter = args.num_iter

        self.sim_mat_conv_spatial = nn.ModuleList(
            [Conv2DBlock((self.emb_dims * 3 + 1, 32, 32), 1) for _ in range(self.num_iter)])
        self.sim_mat_conv_origin = nn.ModuleList([Conv2DBlock((9 + 1, 32, 32), 1) for _ in range(self.num_iter)])
        self.sim_mat_conv = nn.ModuleList([Conv2DBlock((32, 32, 16), 1) for _ in range(self.num_iter)])
        self.sim_mat_conv2 = nn.ModuleList([Conv2DBlock((16, 16, 1), 1) for _ in range(self.num_iter)])
        self.weight_fc = nn.ModuleList([Conv1DBlock((16, 16, 1), 1) for _ in range(self.num_iter)])

        self.head = SVDHead(args=args)

        self.conv_origin_dis = Conv2DBlock((3, 3, 3), 1)
      #  self.conv_origin_dis = Conv2DBlock((3,16, 16),1)
        self.conv_embedding_dis = Conv2DBlock((64, 32, 29), 1)
      #  self.conv_embedding_dis = Conv2DBlock((64,32,32),1)

        self.conv_origin_sm = Conv2DBNReLU(32, 32, 1)
        self.conv_spatial_sm = Conv2DBNReLU(32, 32, 1)
        self.con_for_fuse = Conv2DBlock((32, 32), 1)
        

        self.dis_corr_compatibility = nn.ModuleList([Conv2DBlock((32, 32), 1) for _ in range(self.num_iter)])

    def forward(self, src, tgt, src_embedding, tgt_embedding, src_sig_score, tgt_sig_score, \
                                                                   src_point, tgt_point, src_o_point, tgt_o_point, match_labels):
        ##### initialize #####
        R = torch.eye(3).unsqueeze(0).expand(src.size(0), -1, -1).cuda().float()
        t = torch.zeros(src.size(0), 3).cuda().float()
        ##### initialize #####
        loss = 0.
        num_iter = self.num_iter
        for i in range(num_iter):

            batch_size, num_dims, num_points = src_embedding.size()

            src_dist = src - src_o_point.repeat(1, 1, num_points)
            tgt_dist = tgt - tgt_o_point.repeat(1, 1, num_points)

            src_embedding_dist = src_embedding - src_point.repeat(1, 1, num_points)
            tgt_embedding_dist = tgt_embedding - tgt_point.repeat(1, 1, num_points)

            ##### compute spatial_dis_corr_compatibility #####
            srce_tgte_dis = src_embedding_dist.unsqueeze(-1) - tgt_embedding_dist.unsqueeze(-2)
            spatial_dis_corr_compatibility = self.conv_embedding_dis(srce_tgte_dis)
            ##### compute spatial_dis_corr_compatibility #####

            ##### compute origin_dis_corr_compatibility #####
            src_tgt_dis = src_dist.unsqueeze(-1) - tgt_dist.unsqueeze(-2)
            origin_dis_corr_compatibility = self.conv_origin_dis(src_tgt_dis)
            ##### compute origin_dis_corr_compatibility #####

            ##### compute origin distances #####
            diff_origin = src.unsqueeze(-1) - tgt.unsqueeze(-2)
            dist_origin = (diff_origin ** 2).sum(1, keepdim=True)
            dist_origin = torch.sqrt(dist_origin)
            diff_origin = diff_origin / (dist_origin + 1e-8)
            ##### compute origin distances #####

            ##### compute spatial distances #####
            diff_spatial = src_embedding.unsqueeze(-1) - tgt_embedding.unsqueeze(-2)
            dist_spatial = (diff_spatial ** 2).sum(1, keepdim=True)
            dist_spatial = torch.sqrt(dist_spatial)
            diff_spatial = diff_spatial / (dist_spatial + 1e-8)
            ##### compute spatial distances #####

            ##### similarity spatial matrix convolution to get features #####
            _src_emb = src_embedding.unsqueeze(-1).repeat(1, 1, 1, num_points)
            _tgt_emb = tgt_embedding.unsqueeze(-2).repeat(1, 1, num_points, 1)
            similarity_matrix = torch.cat([_src_emb, _tgt_emb], 1)
            similarity_matrix_spatial = torch.cat((similarity_matrix, diff_spatial, dist_spatial), 1)
            similarity_matrix_spatial = self.sim_mat_conv_spatial[i](similarity_matrix_spatial)
            ##### similarity spatial matrix convolution to get features #####

            ##### similarity origin matrix convolution to get features #####
            _src = src.unsqueeze(-1).repeat(1, 1, 1, num_points)
            _tgt = tgt.unsqueeze(-2).repeat(1, 1, num_points, 1)
            similarity_matrix_origin = torch.cat([_src, _tgt, diff_origin, dist_origin], 1)
            similarity_matrix_origin = self.sim_mat_conv_origin[i](similarity_matrix_origin)
            ##### similarity origin matrix convolution to get features #####

            ##### similarity matrix convolution to get features #####
            similarity_matrix = similarity_matrix_spatial * similarity_matrix_origin
            dis_corr_compatibility = torch.cat((spatial_dis_corr_compatibility, origin_dis_corr_compatibility), dim=1)
            dis_corr_compatibility = self.con_for_fuse(dis_corr_compatibility)
            dis_corr_compatibility = torch.sigmoid(dis_corr_compatibility)
            similarity_matrix = dis_corr_compatibility * similarity_matrix
            similarity_matrix = self.sim_mat_conv[i](similarity_matrix)
            ##### similarity matrix convolution to get features #####

            ##### soft point elimination #####
            weights = similarity_matrix.max(-1)[0]
            weights = self.weight_fc[i](weights).squeeze(1)
            ##### soft point elimination #####

            ##### similarity matrix convolution to get similarities #####
            similarity_matrix = self.sim_mat_conv2[i](similarity_matrix)
            similarity_matrix = similarity_matrix.squeeze(1)
            similarity_matrix = similarity_matrix.clamp(min=-20, max=20)

            ##### similarity matrix convolution to get similarities #####

            ##### negative entropy loss #####
            if self.training and i == 0:
                src_neg_ent = torch.softmax(similarity_matrix, dim=-1)
                src_neg_ent = (src_neg_ent * torch.log(src_neg_ent)).sum(-1)
                tgt_neg_ent = torch.softmax(similarity_matrix, dim=-2)
                tgt_neg_ent = (tgt_neg_ent * torch.log(tgt_neg_ent)).sum(-2)
                neloss = F.mse_loss(src_sig_score, src_neg_ent.detach()) + F.mse_loss(tgt_sig_score,
                                                                                      tgt_neg_ent.detach())
                loss = loss + 0.5 * neloss
            ##### negative entropy loss #####

            ##### matching loss #####
            if self.training:
                temp = torch.softmax(similarity_matrix, dim=-1)
                temp = temp[:, np.arange(temp.size(-2)), np.arange(temp.size(-1))]
                temp = - torch.log(temp)
                match_loss = (temp * match_labels).sum() / match_labels.sum()
                loss = loss + match_loss
                ##### matching loss #####

            ##### finding correspondences #####
            corr_idx = similarity_matrix.max(-1)[1]
            src_corr = tgt[np.arange(tgt.size(0))[:, np.newaxis], :, corr_idx].transpose(1, 2)
            ##### finding correspondences #####

            ##### soft point elimination loss #####
            if self.training:
                weight_labels = (corr_idx == torch.arange(corr_idx.size(1)).cuda().unsqueeze(0)).float()
                num_pos = torch.relu(torch.sum(weight_labels) - 1) + 1
                num_neg = torch.relu(torch.sum(1 - weight_labels) - 1) + 1
                #  weight_loss = F.binary_cross_entropy_with_logits(weights, weight_labels)
                weight_loss = nn.BCEWithLogitsLoss(pos_weight=num_neg * 1.0 / num_pos, reduction='mean')(weights,
                                                                                                         weight_labels.float())
                loss = loss + weight_loss
            ##### soft point elimination loss #####

            ##### hybrid point elimination #####
            weights = torch.sigmoid(weights)
            weights = weights * (weights >= weights.median(-1, keepdim=True)[0]).float()
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-8)
            ##### normalize weights #####

            ##### get R and t #####
            rotation_ab, translation_ab = self.head(src, src_corr, weights)
            rotation_ab = rotation_ab.detach()  # prevent backprop through svd
            translation_ab = translation_ab.detach()  # prevent backprop through svd
            src = torch.matmul(rotation_ab, src) + translation_ab.unsqueeze(-1)
            R = torch.matmul(rotation_ab, R)
            t = torch.matmul(rotation_ab, t.unsqueeze(-1)).squeeze() + translation_ab
            ##### get R and t #####
        euler_ab = npmat2euler(R.detach().cpu().numpy())
        return R, t, euler_ab, loss, src

class ICNET(nn.Module):
    def __init__(self, args):
        super(ICNET, self).__init__()
        self.emb_dims = args.emb_dims
        self.num_iter = args.num_iter
     #   self.traning = training
        self.tau_high = TemperatureNet_high(768)
        self.significance_fc = Conv1DBlock((self.emb_dims, 64, 32, 1), 1)
        self.conv_emb = Conv1DBlock((self.emb_dims * 2+3, 64 * 2, 64), 1)
        self.cross_att = AttentionalPropagation(64, 4)
        self.num_point_preserved = args.num_point_preserved
        self.spe = Conv1DBlock((self.emb_dims, 64, 64), 1)
        self.ope = Conv1DBlock((3, 16, 3), 1)
        self.emb_nn = GNN(args.emb_dims)
        self.dnet = DNET(args=args)

    def forward(self, src, tgt, training, R_gt, t_gt):
        ##### initialize #####
        R = torch.eye(3).unsqueeze(0).expand(src.size(0), -1, -1).cuda().float()
        t = torch.zeros(src.size(0), 3).cuda().float()
        ##### initialize #####
        loss = 0.
        num_iter = self.num_iter
        
         ##### getting ground truth correspondences #####
        if training:
            src_gt = torch.matmul(R_gt, src) + t_gt.unsqueeze(-1)
            dist = src_gt.unsqueeze(-1) - tgt.unsqueeze(-2)
            min_dist, min_idx = (dist ** 2).sum(1).min(-1)  # [B, npoint], [B, npoint]
            dist_for_loss = square_dists(src_gt.transpose(1, 2), tgt.transpose(1, 2))
            min_dist = torch.sqrt(min_dist)
            min_idx = min_idx.cpu().numpy()  # drop to cpu for numpy
            match_labels_real = (min_dist < 0.05).float()
            indicator = match_labels_real.cpu().numpy()
            indicator += 1e-5
            pos_probs = indicator / indicator.sum(-1, keepdims=True)
            indicator = 1 + 1e-5 * 2 - indicator
            neg_probs = indicator / indicator.sum(-1, keepdims=True)
        ##### getting ground truth correspondences #####

        ##### get embedding and significance score #####
        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)
        src_embedding = src_embedding + self.cross_att(src_embedding, tgt_embedding, True)
        tgt_embedding = tgt_embedding + self.cross_att(tgt_embedding, src_embedding, True)

        ##### get tempreature #####
        batch_size, num_dims, num_points = src_embedding.size()
      #  tau_high = self.tau_high(src_embedding, tgt_embedding)
     #   tau_high = tau_high.view(batch_size, num_points, 1)
        batch_idx = np.arange(src.size(0))[:, np.newaxis]
        ##### get tempreature #####

        ##### get similarity_matrix_coarse #####
        similarity_matrix_coarse = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) 
      #  similarity_matrix_coarse = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / tau_high
        similarity_matrix_coarse = torch.softmax(similarity_matrix_coarse, dim=2)
        ##### get similarity_matrix_coarse #####

        ##### spe #####
        max_src_idx, max_tgt_idx = find_most_similar(similarity_matrix_coarse)
        src_point = src_embedding[batch_idx, :, max_src_idx].transpose(1, 2)
        tgt_point = tgt_embedding[batch_idx, :, max_tgt_idx].transpose(1, 2)
        src_embedding_dist = src_embedding - src_point.repeat(1, 1, num_points)
        tgt_embedding_dist = tgt_embedding - tgt_point.repeat(1, 1, num_points)
        src_embedding_pe = self.spe(src_embedding_dist)
        tgt_embedding_pe = self.spe(tgt_embedding_dist)
        ##### spe #####

        ##### ope #####
        src_o_point = src[batch_idx, :, max_src_idx].transpose(1, 2)
        tgt_o_point = tgt[batch_idx, :, max_tgt_idx].transpose(1, 2)
        src_dist = src - src_o_point.repeat(1, 1, num_points)
        tgt_dist = tgt - tgt_o_point.repeat(1, 1, num_points)
        src_pe = self.ope(src_dist)
        tgt_pe = self.ope(tgt_dist)
        ##### ope #####

        ##### get embedding and significance score #####
        src_embedding = torch.cat((src_embedding, src_embedding_pe, src_pe), dim=1)
        tgt_embedding = torch.cat((tgt_embedding, tgt_embedding_pe, tgt_pe), dim=1)
        src_embedding = self.conv_emb(src_embedding)
        tgt_embedding = self.conv_emb(tgt_embedding)
        src_sig_score1 = self.significance_fc(src_embedding).squeeze(1)
        tgt_sig_score1 = self.significance_fc(tgt_embedding).squeeze(1)
        ##### get embedding and significance score #####

        ##### hard point elimination #####
        num_point_preserved = self.num_point_preserved
        if training:
            candidates = np.tile(np.arange(src.size(-1)), (src.size(0), 1))
            pos_idx = batch_choice(candidates, num_point_preserved // 2, p=pos_probs)
            src_idx_real = batch_choice(candidates, num_point_preserved, p=pos_probs)
            neg_idx = batch_choice(candidates, num_point_preserved - num_point_preserved // 2, p=neg_probs)
            src_idx = np.concatenate([pos_idx, neg_idx], 1)
            tgt_idx = min_idx[np.arange(len(src))[:, np.newaxis], src_idx]
            tgt_idx_real = tgt_idx
        else:
            src_idx = src_sig_score1.topk(k=num_point_preserved, dim=-1)[1]
            src_idx = src_idx.cpu().numpy()
            tgt_idx = tgt_sig_score1.topk(k=num_point_preserved, dim=-1)[1]
            tgt_idx = tgt_idx.cpu().numpy()
        if training:
            match_labels = match_labels_real[batch_idx, src_idx]
        src_yuan = src
        src = src[batch_idx, :, src_idx].transpose(1, 2)
        src_embedding = src_embedding[batch_idx, :, src_idx].transpose(1, 2)
        src_sig_score = src_sig_score1[batch_idx, src_idx]
        tgt_yuan = tgt
        tgt = tgt[batch_idx, :, tgt_idx].transpose(1, 2)
        # tgt_key = tgt
        tgt_embedding = tgt_embedding[batch_idx, :, tgt_idx].transpose(1, 2)
        tgt_sig_score = tgt_sig_score1[batch_idx, tgt_idx]
        ##### hard point elimination #####

        if self.training:
            R, t, euler_ab, loss, src_key = self.dnet(src, tgt, src_embedding, tgt_embedding,
                                                                   src_sig_score, tgt_sig_score, \
                                                                   src_point, tgt_point, src_o_point, tgt_o_point,
                                                                   match_labels)
        else:
            R, t, euler_ab, loss, src_key = self.dnet(src, tgt, src_embedding, tgt_embedding,
                                                                   src_sig_score, tgt_sig_score, \
                                                                   src_point, tgt_point, src_o_point, tgt_o_point, None)

        src_idx1 = src_sig_score.topk(k=num_point_preserved, dim=-1)[1]
        tgt_idx1 = tgt_sig_score.topk(k=num_point_preserved, dim=-1)[1]

        src_point = src_yuan[batch_idx, :, src_idx1].transpose(1, 2)
        tgt_point = tgt_yuan[batch_idx, :, tgt_idx1].transpose(1, 2)
        
        src_gt = torch.matmul(R, src) + t.unsqueeze(-1)
        dist = src_gt.unsqueeze(-1) - tgt.unsqueeze(-2)
        min_dist, min_idx = (dist ** 2).sum(1).min(-1)  # [B, npoint], [B, npoint]
        dist_for_loss = square_dists(src_gt.transpose(1, 2), tgt.transpose(1, 2))
        min_dist = torch.sqrt(min_dist)
        min_idx = min_idx.cpu().numpy()  # drop to cpu for numpy
        match_labels_real = (min_dist < 0.05).float()
        indicator = match_labels_real.cpu().numpy()
        indicator += 1e-5
        pos_probs = indicator / indicator.sum(-1, keepdims=True)
        indicator = 1 + 1e-5 * 2 - indicator
        neg_probs = indicator / indicator.sum(-1, keepdims=True)
            
        candidates = np.tile(np.arange(src.size(-1)), (src.size(0), 1))
        pos_idx = batch_choice(candidates, num_point_preserved // 2, p=pos_probs)
        src_idx_real = batch_choice(candidates, num_point_preserved, p=pos_probs)
        neg_idx = batch_choice(candidates, num_point_preserved - num_point_preserved // 2, p=neg_probs)
        src_idx = np.concatenate([pos_idx, neg_idx], 1)
        tgt_idx = min_idx[np.arange(len(src))[:, np.newaxis], src_idx]
        tgt_idx_real = tgt_idx
        
        src_key_real = src_yuan[batch_idx, :, src_idx_real].transpose(1, 2)
        tgt_key_real = tgt_yuan[batch_idx, :, tgt_idx_real].transpose(1, 2) 
               
        
        return R, t, euler_ab, loss, src_key_real, tgt_key_real, src_key, tgt

        

class MODEL(nn.Module):
    def __init__(self, args):
        super(MODEL, self).__init__()
        self.emb_dims = args.emb_dims
        
        self.icnet = ICNET(args=args)


    def get_state(self):
        return self.icnet.state_dict()



    def forward(self, src, tgt, R_gt=None, t_gt=None):

        ##### only pass ground truth while training #####
        if not (self.training or (R_gt is None and t_gt is None)):
            raise Exception('Passing ground truth while testing')
        ##### only pass ground truth while training #####        
        R, t, euler_ab, loss, src_key_real, tgt_key_real, src_key, tgt = self.icnet(src, tgt, self.training, R_gt, t_gt)
       
        return R, t, loss, euler_ab, src_key_real, tgt_key_real, src_key, tgt