import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .blocks import DBlock
from clrnet.models.heads.transformer import build_transformer
from mmcv.cnn import ConvModule
from ..registry import NECKS


class PagFMUP(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
        super(PagFMUP, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels,
                          kernel_size=1, bias=False),
                BatchNorm(in_channels)
            )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))

        y = F.interpolate(y, size=[input_size[2], input_size[3]],
                          mode='bilinear', align_corners=False)
        x = (1 - sim_map) * x + sim_map * y

        return x
class PagFMDown(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
        super(PagFMDown, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels,
                          kernel_size=1, bias=False),
                BatchNorm(in_channels)
            )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)
        self.Done = DownSamplerBlock(64,64)
        self.Done2 = DownSamplerBlock(64, 64)
    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        y_q = self.f_y(y)
        y_q = self.Done(y_q)
        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))

        y = self.Done2(y)
        x = (1 - sim_map) * x + sim_map * y

        return x
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self,
                 num_pos_feats=64,
                 temperature=10000,
                 normalize=False,
                 scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

def build_position_encoding(hidden_dim, shape):
    mask = torch.zeros(shape, dtype=torch.bool)
    pos_module = PositionEmbeddingSine(hidden_dim // 2)
    pos_embs = pos_module(mask)
    return pos_embs
class DownSamplerBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(in_channel*2, in_channel, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(in_channel, out_channel, 1, bias=False)
        )
        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=2,padding=1)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-3, track_running_stats=True)

    def forward(self, input):
        output1 = self.conv1(input)
        maxout = self.max_pool(input)
        avgout = self.avg_pool(input)
        output2 = self.mlp(torch.cat((maxout,avgout),dim=1))
        out = F.relu(self.bn(output1+output2))

        return out
class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput, up_width, up_height):
        super().__init__()

        self.conv = nn.ConvTranspose2d(
            ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)

        self.bn = nn.BatchNorm2d(noutput, eps=1e-3, track_running_stats=True)



        # interpolate
        self.up_width = up_width
        self.up_height = up_height
        self.interpolate_conv = nn.Conv2d(ninput, noutput,kernel_size=1)
        self.interpolate_bn = nn.BatchNorm2d(
            noutput, eps=1e-3, track_running_stats=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        out = F.relu(output)

        interpolate_output = self.interpolate_conv(input)
        interpolate_output = self.interpolate_bn(interpolate_output)
        interpolate_output = F.relu(interpolate_output)

        interpolate = F.interpolate(interpolate_output, size=[self.up_height,  self.up_width],
                                    mode='bilinear', align_corners=False)

        return out + interpolate


def build_position_encoding(hidden_dim, shape):
    mask = torch.zeros(shape, dtype=torch.bool)
    pos_module = PositionEmbeddingSine(hidden_dim // 2)
    pos_embs = pos_module(mask)
    return pos_embs


class AttentionLayer(nn.Module):
    """ Position attention module"""

    def __init__(self, in_dim, out_dim, ratio=4, stride=1):
        super(AttentionLayer, self).__init__()
        self.chanel_in = in_dim
        norm_cfg = dict(type='BN', requires_grad=True)
        act_cfg = dict(type='ReLU')
        self.pre_conv = ConvModule(
            in_dim,
            out_dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)
        self.query_conv = nn.Conv2d(
            in_channels=out_dim, out_channels=out_dim // ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=out_dim, out_channels=out_dim // ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=out_dim, out_channels=out_dim, kernel_size=1)
        self.final_conv = ConvModule(
            out_dim,
            out_dim,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, pos=None):
        """
            inputs :
                x : inpput feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        x = self.pre_conv(x)
        m_batchsize, _, height, width = x.size()
        if pos is not None:
            x += pos
        proj_query = self.query_conv(x).view(m_batchsize, -1,
                                             width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        attention = attention.permute(0, 2, 1)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention)
        out = out.view(m_batchsize, -1, height, width)
        proj_value = proj_value.view(m_batchsize, -1, height, width)
        out_feat = self.gamma * out + x
        out_feat = self.final_conv(out_feat)
        return out_feat


class TransConvEncoderModule(nn.Module):
    def __init__(self, in_dim, attn_in_dims, attn_out_dims, strides, ratios, downscale=True, pos_shape=None):
        super(TransConvEncoderModule, self).__init__()
        if downscale:
            stride = 2
        else:
            stride = 1
        # self.first_conv = ConvModule(in_dim, 2*in_dim, kernel_size=3, stride=stride, padding=1)
        # self.final_conv = ConvModule(attn_out_dims[-1], attn_out_dims[-1], kernel_size=3, stride=1, padding=1)
        attn_layers = []
        for dim1, dim2, stride, ratio in zip(attn_in_dims, attn_out_dims, strides, ratios):
            attn_layers.append(AttentionLayer(dim1, dim2, ratio, stride))
        if pos_shape is not None:
            self.attn_layers = nn.ModuleList(attn_layers)
        else:
            self.attn_layers = nn.Sequential(*attn_layers)
        self.pos_shape = pos_shape
        self.pos_embeds = []
        if pos_shape is not None:
            for dim in attn_out_dims:
                pos_embed = build_position_encoding(dim, pos_shape).cuda()
                self.pos_embeds.append(pos_embed)

    def forward(self, src):
        # src = self.first_conv(src)
        if self.pos_shape is None:
            src = self.attn_layers(src)
        else:
            for layer, pos in zip(self.attn_layers, self.pos_embeds):
                src = layer(src, pos.to(src.device))
        # src = self.final_conv(src)
        return src
@NECKS.register_module
class LKFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 attention=False,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(type='Xavier',
                               layer='Conv2d',
                               distribution='uniform'),
                 cfg=None):
        super(LKFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.dblock = DBlock(64, 64, [1], 16, 2, attention='se')
        self.num_outs = num_outs
        self.attention = attention
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs

        self.transformer = TransConvEncoderModule(64,[64,256],[256,64],[1,1],[4,4],[24,20,50])
        self.transformer0 = TransConvEncoderModule(64, [64, 256], [256, 64], [1, 1], [4, 4], [24, 10, 25])
        self.transformer1 = TransConvEncoderModule(64, [64, 256], [256, 64], [1, 1], [4, 4], [24, 40, 100])


        self.w1 = nn.Parameter(torch.full((1,), 0.5))
        self.w2 = nn.Parameter(torch.full((1,), 0.5))
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.downsample = DownSamplerBlock(64,64)
        self.dblock2 = DBlock(64, 64, [1], 16, 2, attention='se')
        self.downsample2 = DownSamplerBlock(64,64)
        self.fpn_convs = nn.ModuleList()
        self.upsample = UpsamplerBlock(64,64,50,20)
        self.upsample2 = UpsamplerBlock(64,64,100,40)
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            fpn_conv = ConvModule(out_channels,
                                  out_channels,
                                  3,
                                  padding=1,
                                  conv_cfg=conv_cfg,
                                  norm_cfg=norm_cfg,
                                  act_cfg=act_cfg,
                                  inplace=False)
            self.fpn_convs.append(fpn_conv)
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(in_channels,
                                            out_channels,
                                            3,
                                            stride=2,
                                            padding=1,
                                            conv_cfg=conv_cfg,
                                            norm_cfg=norm_cfg,
                                            act_cfg=act_cfg,
                                            inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) >= len(self.in_channels)

        if len(inputs) > len(self.in_channels):
            for _ in range(len(inputs) - len(self.in_channels)):
                del inputs[0]

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        out = self.w1*(self.downsample(laterals[0]))+laterals[1]+self.w2*(self.upsample(laterals[-1]))
        out1 = self.downsample2(out)
        out2 = self.upsample2(out)
        result = [out1,out,out2]
        #out = self.fpn_convs[1](out)
        # list0 = self.transformer(out)
        # list2 = self.transformer0(out1)
        # list1 = self.transformer1(out2)
        #result = [list1,list0,list2]
        #pos = build_position_encoding(64,(24,20,50))

        # out0 = self.downsample2(out1)+laterals[2]
        # out2 = self.upsample2(out1)+laterals[0]
        # outs = [out2,out1,out0]
        #
        used_backbone_levels = len(laterals)
        # outs = [
        #     self.fpn_convs[i](outs[i]) for i in range(used_backbone_levels)
        # ]
        outs = [
            self.fpn_convs[i](result[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return outs


