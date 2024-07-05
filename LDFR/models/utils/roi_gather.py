import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        #attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        #attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        #attn_2 = self.conv2_2(attn_2)
        attn = attn_0 + attn_1 + attn_2+attn

        attn = self.conv3(attn)

        return attn

def LinearModule(hidden_dim):
    return nn.ModuleList(
        [nn.Linear(hidden_dim, hidden_dim),
         nn.ReLU(inplace=True)])


class FeatureResize(nn.Module):
    def __init__(self, size=(10, 25)):
        super(FeatureResize, self).__init__()
        self.size = size

    def forward(self, x):
        x = F.interpolate(x, self.size)
        return x.flatten(2)


class ROIGather(nn.Module):
    '''
    ROIGather module for gather global information
    Args: 
        in_channels: prior feature channels
        num_priors: prior numbers we predefined
        sample_points: the number of sampled points when we extract feature from line
        fc_hidden_dim: the fc output channel
        refine_layers: the total number of layers to build refine
    '''
    def __init__(self,
                 in_channels,
                 num_priors,
                 sample_points,
                 fc_hidden_dim,
                 refine_layers,
                 mid_channels=48):
        super(ROIGather, self).__init__()
        self.in_channels = in_channels
        self.num_priors = num_priors
        self.f_key = ConvModule(in_channels=self.in_channels,
                                out_channels=self.in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                norm_cfg=dict(type='BN'))

        self.f_query = nn.Sequential(
            nn.Conv1d(in_channels=num_priors,
                      out_channels=num_priors,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      groups=num_priors),
            nn.ReLU(),
        )
        self.f_value = nn.Conv2d(in_channels=self.in_channels,
                                 out_channels=self.in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.W = nn.Conv1d(in_channels=num_priors,
                           out_channels=num_priors,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=num_priors)

        self.resize = FeatureResize()
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.convs = nn.ModuleList()
        self.catconv = nn.ModuleList()
        for i in range(refine_layers):
            self.convs.append(
                ConvModule(in_channels,
                           mid_channels, (9, 1),
                           padding=(4, 0),
                           bias=False,
                           norm_cfg=dict(type='BN')))
            # self.convs.append(
            #     ConvModule(in_channels,
            #                mid_channels, (7, 1),
            #                padding=(3, 0),
            #                bias=False,
            #                norm_cfg=dict(type='BN')))
            # self.convs.append(
            #     ConvModule(in_channels,
            #                mid_channels, (11, 1),
            #                padding=(5, 0),
            #                bias=False,
            #                norm_cfg=dict(type='BN')))


            self.catconv.append(
                ConvModule(mid_channels * (i + 1),
                           in_channels, (9, 1),
                           padding=(4, 0),
                           bias=False,
                           norm_cfg=dict(type='BN')))
        self.Convx = AttentionModule(64)
        self.fc = nn.Linear(sample_points * fc_hidden_dim, fc_hidden_dim)

        self.fc_norm = nn.LayerNorm(fc_hidden_dim)

    def roi_fea(self, x, layer_index):
        feats = []
        for i, feature in enumerate(x):
            feat_trans1 = self.convs[i](feature)
            # feat_trans2 = self.convs[i+1](feature)
            # feat_trans3 = self.convs[i+2](feature)
            # feat_trans = feat_trans1+feat_trans2+feat_trans3
            feats.append(feat_trans1)
        cat_feat = torch.cat(feats, dim=1)
        cat_feat = self.catconv[layer_index](cat_feat)
        return cat_feat

    def forward(self, roi_features, x, layer_index):
        '''
        Args:
            roi_features: prior feature, shape: (Batch * num_priors, prior_feat_channel, sample_point, 1)
            x: feature map
            layer_index: currently on which layer to refine
        Return: 
            roi: prior features with gathered global information, shape: (Batch, num_priors, fc_hidden_dim)
        '''
        roi = self.roi_fea(roi_features, layer_index)
        bs = x.size(0)
        xk = self.Convx(x)
        roi = roi.contiguous().view(bs * self.num_priors, -1)

        roi = F.relu(self.fc_norm(self.fc(roi)))
        roi = roi.view(bs, self.num_priors, -1)
        query = roi

        value = self.resize(self.f_value(x))
        query = self.f_query(query)
        key = self.f_key(xk)
        value = value.permute(0, 2, 1)
        key = self.resize(key)
        sim_map = torch.matmul(query, key)
        sim_map = (self.in_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = self.W(context)

        roi = roi + F.dropout(context, p=0.1, training=self.training)

        return roi
