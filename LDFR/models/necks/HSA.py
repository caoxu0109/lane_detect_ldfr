
import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

class ALU(nn.Module):
    def __init__(self,in_channel,out_channel=64, reduction=16, spatial_kernel=7,inplace=False):
        super(ALU,self).__init__()
        self.conv1 = nn.Conv2d(in_channel,
                                  out_channel,
                                  1,
                                  stride=1,
                                padding=0)
        # self.conv2 = nn.Conv2d(in_channel,
        #                           out_channel,
        #                           3,
        #                           stride=1,
        #                         padding=1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(in_channel, in_channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(in_channel // reduction, out_channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self,x):
        x1 = self.conv1(x)
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x2 = channel_out * x1

        max_out, _ = torch.max(x2, dim=1, keepdim=True)
        # print('max_out:',max_out.shape)
        avg_out = torch.mean(x2, dim=1, keepdim=True)
        # print('avg_out:',avg_out.shape)
        a = torch.cat([max_out, avg_out], dim=1)
        # print('a:',a.shape)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        # print('spatial:',spatial_out.shape)
        x3 = spatial_out * x2

        out = x3 + x1

        # out = self.conv2(out)
        return out
def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class PSA_p(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PSA_p, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size-1)//2

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)

        batch, channel, height, width = input_x.size()

        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)

        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)

        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1,2))
        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]
        context = self.conv_up(context)

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out

    def channel_pool(self, x):
        # [N, IC, H, W]
        g_x = self.conv_q_left(x)

        batch, channel, height, width = g_x.size()

        # [N, IC, 1, 1]
        avg_x = self.avg_pool(g_x)

        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        # [N, 1, IC]
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        # [N, IC, H*W]
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)

        # [N, 1, H*W]
        # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
        context = torch.matmul(avg_x, theta_x)
        # [N, 1, H*W]
        context = self.softmax_left(context)

        # [N, 1, H, W]
        context = context.view(batch, 1, height, width)

        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)

        out = x * mask_sp

        return out

    def forward(self, x):
        # [N, C, H, W]
        context_channel = self.spatial_pool(x)
        # [N, C, H, W]
        context_spatial = self.channel_pool(x)
        # [N, C, H, W]
        out = context_spatial + context_channel
        return out
#PSA模块究竟是做咩的
class PSA_s(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PSA_s, self).__init__()

        self.inplanes = inplanes#输入通道数
        self.inter_planes = planes // 2#中间通道数
        self.planes = planes#输出通道数
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        ratio = 4
        #1*1的卷积把通道数降为1
        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        #1*1的卷积变换通道到中间通道数
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                      bias=False)
        # self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        #分两次1*1的卷积,一次先变成输入通道的1/4,第二次卷积到输出通道数
        self.conv_up = nn.Sequential(
            nn.Conv2d(self.inter_planes, self.inter_planes // ratio, kernel_size=1),
            nn.LayerNorm([self.inter_planes // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_planes // ratio, self.planes, kernel_size=1)
        )
        #在dim=2通道层面进行softmax
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True
    #空间下采样  结合空间像素的值获取通道注意力
    def spatial_pool(self, x):
        ##1*1的卷积变换通道到中间通道数
        input_x = self.conv_v_right(x)

        batch, channel, height, width = input_x.size()

        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]  #用1*1的卷积把通道数降为1
        context_mask = self.conv_q_right(x)

        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)

        # [N, 1, H*W] #在dim=2层面进行softmax
        context_mask = self.softmax_right(context_mask)

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1, 2))

        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]#分两次1*1的卷积,一次先变成输入通道的1/4,第二次卷积到输出通道数
        context = self.conv_up(context)

        # [N, OC, 1, 1]#进行sigmoid
        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out
    #用通道给全局像素进行注意力加权
    def channel_pool(self, x):
        # [N, IC, H, W] 用1*1卷积将通道变为inter planes
        g_x = self.conv_q_left(x)

        batch, channel, height, width = g_x.size()

        # [N, IC, 1, 1] 全局平均池化
        avg_x = self.avg_pool(g_x)

        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        # [N, 1, IC]
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        # [N, IC, H*W]
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)

        # [N, IC, H*W]
        theta_x = self.softmax_left(theta_x)

        # [N, 1, H*W]
        # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
        context = torch.matmul(avg_x, theta_x)

        # [N, 1, H, W]
        context = context.view(batch, 1, height, width)

        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)

        out = x * mask_sp

        return out

    def forward(self, x):
        # [N, C, H, W]
        out = self.spatial_pool(x)

        # [N, C, H, W]
        out = self.channel_pool(out)

        # [N, C, H, W]
        # out = context_spatial + context_channel

        return out

class UPx2(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''
    def __init__(self, nIn, nOut):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        self.deconv = nn.ConvTranspose2d(nIn, nOut, (2,1), stride=(2,1), padding=(0,0))
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.deconv(input)
        output = self.bn(output)
        output = self.act(output)
        return output
    def fuseforward(self, input):
        output = self.deconv(input)
        output = self.act(output)
        return output

class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.conv_q = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.conv_k = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.conv_v = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        if x.shape[2]==10:
            fds = self.conv1(x)
        else:
            fds = self.conv1(x)
            fds = self.conv2(fds)
        Q = self.conv_q(fds)#.transpose(1,2)
        K = self.conv_k(fds)
        V = self.conv_v(fds)
        A = torch.matmul(Q.transpose(2,3), K)
        A = F.softmax(A, dim=-1)
        fo = torch.matmul(V, A)
        return fo#.transpose(1,2)
class Att_mechanism(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Att_mechanism, self).__init__()
        self.attention_module = AttentionModule(in_channels, out_channels)
        self.Up = UPx2(out_channels,out_channels)

    def forward(self, x):
        fo = self.attention_module(x)
        if fo.shape[2] == 5:
            fo = self.Up(fo)
        else:
            fo = self.Up(fo)
            fo = self.Up(fo)
        # 这里可以添加上采样操作，以便与原始输入特征图对齐
        return fo

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        # conv则是实际进行的卷积操作，注意这里步长设置为卷积核大小，因为与该卷积核进行卷积操作的特征图是由输出特征图中每个点扩展为其对应卷积核那么多个点后生成的。
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        # p_conv是生成offsets所使用的卷积，输出通道数为卷积核尺寸的平方的2倍，代表对应卷积核每个位置横纵坐标都有偏移量。
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation  # modulation是可选参数,若设置为True,那么在进行卷积操作时,对应卷积核的每个位置都会分配一个权重。
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        # 由于卷积核中心点位置是其尺寸的一半，于是中心点向左（上）方向移动尺寸的一半就得到起始点，向右（下）方向移动另一半就得到终止点
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        # p0_y、p0_x就是输出特征图每点映射到输入特征图上的纵、横坐标值。
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    # 输出特征图上每点（对应卷积核中心）加上其对应卷积核每个位置的相对（横、纵）坐标后再加上自学习的（横、纵坐标）偏移量。
    # p0就是将输出特征图每点对应到卷积核中心，然后映射到输入特征图中的位置；
    # pn则是p0对应卷积核每个位置的相对坐标；
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        # 计算双线性插值点的4邻域点对应的权重
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset

class FlipAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels*2,channels,kernel_size=1)
        self.Key = nn.Conv2d(channels, channels, kernel_size=1)
        self.Value = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, feature,flipped):
        concat = torch.cat([feature, flipped], dim=1)
        Q = self.query(concat)
        K = self.Key(feature)
        V = self.Value(feature)
        att = torch.matmul(Q.transpose(2,3), K)
        att = F.softmax(att,dim=-1)
        out = torch.matmul(V,att)
        return feature + out


class FeatureFlipatt(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(channels)
        )
        self.proj2_conv = FlipAttention(channels)
        self.DeformConv2d = DeformConv2d(channels,channels)
        self.proj2_norm = nn.BatchNorm2d(channels)

    def forward(self, feature):
        # B x C x H x W
        flipped = feature.flip(-1)  # An auto-copy

        feature = self.proj1(feature)
        flipped = self.proj2_conv(flipped, feature)

       # flipped = self.DeformConv2d(flipped)
        flipped = self.proj2_norm(flipped)

        return feature + flipped

class RESA(nn.Module):
    def __init__(self, channel,h,w):
        super(RESA, self).__init__()
        self.iter = 4
        chan = channel
        self.height = h
        self.width = w
        self.alpha = 0.25
        conv_stride = 9

        for i in range(self.iter):
            conv_vert1 = nn.Conv2d(
                chan, chan, (1, conv_stride),
                padding=(0, conv_stride//2), groups=1, bias=False)
            conv_vert2 = nn.Conv2d(
                chan, chan, (1, conv_stride),
                padding=(0, conv_stride//2), groups=1, bias=False)

            setattr(self, 'conv_d'+str(i), conv_vert1)
            setattr(self, 'conv_u'+str(i), conv_vert2)

            conv_hori1 = nn.Conv2d(
                chan, chan, (conv_stride, 1),
                padding=(conv_stride//2, 0), groups=1, bias=False)
            conv_hori2 = nn.Conv2d(
                chan, chan, (conv_stride, 1),
                padding=(conv_stride//2, 0), groups=1, bias=False)

            setattr(self, 'conv_r'+str(i), conv_hori1)
            setattr(self, 'conv_l'+str(i), conv_hori2)

            idx_d = (torch.arange(self.height) + self.height //
                     2**(self.iter - i)) % self.height
            setattr(self, 'idx_d'+str(i), idx_d)

            idx_u = (torch.arange(self.height) - self.height //
                     2**(self.iter - i)) % self.height
            setattr(self, 'idx_u'+str(i), idx_u)

            idx_r = (torch.arange(self.width) + self.width //
                     2**(self.iter - i)) % self.width
            setattr(self, 'idx_r'+str(i), idx_r)

            idx_l = (torch.arange(self.width) - self.width //
                     2**(self.iter - i)) % self.width
            setattr(self, 'idx_l'+str(i), idx_l)

    def forward(self, x):
        x = x.clone()

        for direction in ['d', 'u']:
            for i in range(self.iter):
                conv = getattr(self, 'conv_' + direction + str(i))
                idx = getattr(self, 'idx_' + direction + str(i))
                y = x[..., idx, :]
                x.add_(self.alpha * F.relu(conv(x[..., idx, :])))

        for direction in ['r', 'l']:
            for i in range(self.iter):
                conv = getattr(self, 'conv_' + direction + str(i))
                idx = getattr(self, 'idx_' + direction + str(i))
                x.add_(self.alpha * F.relu(conv(x[..., idx])))

        return x