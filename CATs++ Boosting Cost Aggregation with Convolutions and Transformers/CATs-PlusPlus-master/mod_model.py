import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from functools import reduce
from operator import add
from contextlib import nullcontext
from torchvision.models import resnet
from torchvision.models import vgg
from torch.nn.modules.normalization import GroupNorm
from timm.models.layers import trunc_normal_

try:
    import clip
except:
    print('CLIP is not installed! "pip install git+https://github.com/openai/CLIP.git" to use CLIP backbone.')

class CATsImproved(nn.Module):
    def __init__(self, backbone='resnet101', freeze=False):
        super().__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=True)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = self.extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = self.extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = self.extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
        elif backbone == 'clip_resnet101':
            self.backbone = clip.load("RN101")[0].float()
            self.feat_ids = list(range(4, 34))
            self.extract_feats = self.extract_feat_clip
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.backbone.eval()
        self.hpn_learner = CATs(inch=list(reversed(nbottlenecks[-3:])))
        self.freeze = freeze

    def extract_feat_vgg(self, img, backbone, feat_ids, bottleneck_ids=None, lids=None):
        """Extract intermediate features from VGG"""
        feats = []
        feat = img
        for lid, module in enumerate(backbone.features):
            feat = module(feat)
            if lid in feat_ids:
                feats.append(feat.clone())
        return feats

    def extract_feat_res(self, img, backbone, feat_ids, bottleneck_ids, lids):
        """Extract intermediate features from ResNet"""
        feats = []
        # Layer 0
        feat = backbone.conv1.forward(img)
        feat = backbone.bn1.forward(feat)
        feat = backbone.relu.forward(feat)
        feat = backbone.maxpool.forward(feat)
        # Layer 1-4
        for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
            res = feat
            feat = backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
            feat = backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
            feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
            feat = backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
            feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
            feat = backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

            if bid == 0:
                res = backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)
            
            feat += res

            if hid + 1 in feat_ids:
                feats.append(feat.clone())

            feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        return feats

    def stack_feats(self, feats):
        feats_l4 = torch.stack(feats[-self.stack_ids[0]:]).transpose(0, 1)
        feats_l3 = torch.stack(feats[-self.stack_ids[1]:-self.stack_ids[0]]).transpose(0, 1)
        feats_l2 = torch.stack(feats[-self.stack_ids[2]:-self.stack_ids[1]]).transpose(0, 1)
        return [feats_l4, feats_l3, feats_l2]
    
    def multilayer_correlation(self, query_feats, support_feats, stack_ids):
        """Computes multi-layer correlation"""
        eps = 1e-5

        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            bsz, ch, hb, wb = support_feat.size()
            support_feat = support_feat.view(bsz, ch, -1)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)

            bsz, ch, ha, wa = query_feat.size()
            query_feat = query_feat.view(bsz, ch, -1)
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)

            corr = torch.bmm(query_feat.transpose(1, 2), support_feat).view(bsz, ha, wa, hb, wb)
            corr = corr.clamp(min=0)
            corrs.append(corr)

        corr_l4 = torch.stack(corrs[-stack_ids[0]:]).transpose(0, 1).contiguous()
        corr_l3 = torch.stack(corrs[-stack_ids[1]:-stack_ids[0]]).transpose(0, 1).contiguous()
        corr_l2 = torch.stack(corrs[-stack_ids[2]:-stack_ids[1]]).transpose(0, 1).contiguous()

        return [corr_l4, corr_l3, corr_l2]
    
    def forward(self, trg_img, src_img):
        with torch.no_grad() if self.freeze else nullcontext():
            trg_feats = self.extract_feats(trg_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            src_feats = self.extract_feats(src_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)

        corr = self.multilayer_correlation(trg_feats, src_feats, self.stack_ids)
        flow = self.hpn_learner(corr, self.stack_feats(trg_feats), self.stack_feats(src_feats))
        return flow







class MaxPool4d(nn.Module):
    def __init__(self, kernel_size, stride, padding, dim='support'):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride, padding, ceil_mode=True)
        self.dim = dim
        
    def forward(self, x):
        """
        x: Hyper correlation.
            shape: B L H_q W_q H_s W_s
        """
        B, L, H_q, W_q, H_s, W_s = x.size()
        
        if self.dim == 'support':
            x = rearrange(x, 'B L H_q W_q H_s W_s -> (B H_q W_q) L H_s W_s')
            x = self.pool(x)
            x = rearrange(x, '(B H_q W_q) L H_s W_s -> B L H_q W_q H_s W_s', H_q=H_q, W_q=W_q)
        elif self.dim == 'query':
            x = rearrange(x, 'B L H_q W_q H_s W_s -> (B H_s W_s) L H_q W_q')
            x = self.pool(x)
            x = rearrange(x, '(B H_s W_s) L H_q W_q -> B L H_q W_q H_s W_s', H_s=H_s, W_s=W_s)
        else:
            raise NotImplemented(f'Invalid dimension {self.dim}. dim should be "support" or "query"')
        return x
    

class Interpolate4d_for_conv4d(nn.Module):
    def __init__(self, size, dim='support'):
        super().__init__()
        self.size = size
        self.dim = dim
        
    def forward(self, x):
        """
        x: Hyper correlation.
        """
        B, L, H_q, W_q, H_s, W_s = x.size()
        if self.dim == 'support':
            x = rearrange(x, 'B L H_q W_q H_s W_s -> (B H_q W_q) L H_s W_s')
            x = F.interpolate(x, size=self.size, mode='bilinear', align_corners=True)
            x = rearrange(x, '(B H_q W_q) L H_s W_s -> B L H_q W_q H_s W_s', H_q=H_q, W_q=W_q)
        elif self.dim == 'query':
            x = rearrange(x, 'B L H_q W_q H_s W_s -> (B H_s W_s) L H_q W_q')
            x = F.interpolate(x, size=self.size, mode='bilinear', align_corners=True)
            x = rearrange(x, '(B H_s W_s) L H_q W_q -> B L H_q W_q H_s W_s', H_s=H_s, W_s=W_s)
        else:
            raise NotImplemented(f'Invalid dimension {self.dim}. dim should be "support" or "query"')
        return x
        

class Conv4d(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=True,
            transposed_query=False,
            transposed_supp=False,
            target_size=None,
            output_padding=None
        ):
        super().__init__()
        
        if transposed_query:
            assert output_padding is not None, 'output_padding cannot be None'
            self.query_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size[:2], stride=stride[:2],
                bias=bias, padding=padding[:2], output_padding=output_padding[:2])
        else:
            self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size[:2], stride=stride[:2],
                bias=bias, padding=padding[:2])
            
        if transposed_supp:
            assert output_padding is not None, 'output_padding cannot be None'
            self.supp_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size[2:], stride=stride[2:],
                bias=bias, padding=padding[2:], output_padding=output_padding[2:])
        else:
            self.supp_conv = nn.Conv2d(in_channels, out_channels, kernel_size[2:], stride=stride[2:],
                bias=bias, padding=padding[2:])
            
        self.change_supp = stride[-1] > 1 or stride[0] == 1 and kernel_size[0] == 1
        if self.change_supp:
            if transposed_supp:
                assert target_size is not None, 'Invalid size'
                self.pool_supp = Interpolate4d_for_conv4d(target_size[-2:], dim='support')
            else:
                self.pool_supp = MaxPool4d(kernel_size=stride[-2:], stride=stride[-2:], padding=(0, 0), dim='support')
        
        self.change_query = stride[0] > 1 or stride[0] == 1 and kernel_size[0] == 1
        if self.change_query:
            if transposed_query:
                assert target_size is not None, 'Invalid size'
                self.pool_query = Interpolate4d_for_conv4d(target_size[:2], dim='query')
            else:
                self.pool_query = MaxPool4d(kernel_size=stride[:2], stride=stride[:2], padding=(0, 0), dim='query')
            
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def forward(self, x):
        """
        x: Hyper correlation map.
            shape: B L H_q W_q H_s W_s
        """
        B, L, H_q, W_q, H_s, W_s = x.size()
        
        if self.change_supp:
            x_query = self.pool_supp(x)
            H_s, W_s = x_query.shape[-2:]
        else:
            x_query = x.clone()
        
        if self.change_query:
            x_supp = self.pool_query(x)
            H_q, W_q = x_supp.shape[2:4]
        else:
            x_supp = x.clone()
        
        x_query = rearrange(x_query, 'B L H_q W_q H_s W_s -> (B H_s W_s) L H_q W_q')
        x_query = self.query_conv(x_query)
        x_query = rearrange(x_query, '(B H_s W_s) L H_q W_q -> B L H_q W_q H_s W_s', H_s=H_s, W_s=W_s)
        
        x_supp = rearrange(x_supp, 'B L H_q W_q H_s W_s -> (B H_q W_q) L H_s W_s')
        x_supp = self.supp_conv(x_supp)
        x_supp = rearrange(x_supp, '(B H_q W_q) L H_s W_s -> B L H_q W_q H_s W_s', H_q=H_q, W_q=W_q)
        
        return x_query + x_supp


class Encoder4D(nn.Module):
    def __init__(self,
        corr_levels,
        kernel_size,
        stride,
        padding,
        group=(4,),
    ):
        super().__init__()
        self.conv4d = nn.ModuleList([])
        for i, (k, s, p) in enumerate(zip(kernel_size, stride, padding)):
            conv4d = nn.Sequential(
                Conv4d(corr_levels[i], corr_levels[i + 1], k, s, p),
                nn.GroupNorm(group[i], corr_levels[i + 1]),
                nn.ReLU() # No inplace for residual
            )
            self.conv4d.append(conv4d)
        
    def forward(self, x):
        """
        x: Hyper correlation. B L H_q W_q H_s W_s
        """
        for conv in self.conv4d:
            x = conv(x)
        # Patch embedding for transformer
        return x

class Interpolate4d(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
    
    def forward(self, x):
        B, C, H_t, W_t, _, _ = x.shape
        x = rearrange(x, 'B C H_t W_t H_s W_s -> B (C H_t W_t) H_s W_s')
        x = F.interpolate(x, size=self.size[-2:], mode='bilinear', align_corners=True)
        x = rearrange(x, 'B (C H_t W_t) H_s W_s -> B (C H_s W_s) H_t W_t', C=C, H_t=H_t, W_t=W_t)
        x = F.interpolate(x, size=self.size[:2], mode='bilinear', align_corners=True)
        x = rearrange(x, 'B (C H_s W_s) H_t W_t -> B C H_t W_t H_s W_s', C=C, H_s=self.size[-2], W_s=self.size[-1])
        return x

def transpose4d(x):
    x = rearrange(x, 'B C H_t W_t H_s W_s -> B C H_s W_s H_t W_t')
    return x



# Transformer definition (from transformer.py)
class Transformer(nn.Module):
    def __init__(self, in_channel, depth=2, affinity_dim=64, target_proj_dim=384, nhead=4, mlp_ratio=4.,
                    input_corr_size=(8, 8, 8, 8), kernel_size=(3, 3, 3, 3), stride=(2, 2, 1, 1), padding=(1, 1, 1, 1), group=1):
        super().__init__()
        assert stride[-1] == 1 and stride[-2] == 1, 'stride of source dimension must be 1'

        self.layer = nn.ModuleList([
            TransformerLayer(in_channel, affinity_dim, target_proj_dim, nhead, mlp_ratio, input_corr_size, kernel_size, stride, padding, group)
            for _ in range(depth)
        ])

    def forward(self, x, affinity):
        for layer in self.layer:
            x = layer(x, affinity)

        return x

# TransformerLayer definition (from transformer.py)
class TransformerLayer(nn.Module):
    def __init__(self, in_channel, affinity_dim=64, target_proj_dim=384, nhead=4, mlp_ratio=4., 
        input_corr_size=(8, 8, 8, 8), kernel_size=(3, 3, 3, 3), stride=(2, 2, 1, 1), padding=(1, 1, 1, 1), group=1):
        super().__init__()
        self.nhead = nhead
        self.qk = LinearConv4d(in_channel, in_channel, affinity_dim, target_proj_dim * 2, input_corr_size, kernel_size, stride, padding, group)
        self.v = nn.Sequential(
            Conv4d(
                in_channels=in_channel,
                out_channels=in_channel,
                kernel_size=kernel_size,
                stride=(1, 1, 1, 1),
                padding=padding,
            ),
            nn.GroupNorm(group, in_channel)
        )
        self.attn = SelfAttention()
        self.mlp = MLPConv4d(in_channel, mlp_ratio, kernel_size, stride=(1, 1, 1, 1), padding=padding)
        
        self.norm1 = GroupNorm(group, in_channel)
        self.norm2 = GroupNorm(group, in_channel)

        self.pos_embed = nn.Parameter(torch.zeros(1, input_corr_size[-2] * input_corr_size[-1], 1, target_proj_dim // nhead))
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x, affinity):
        """
        x: B C H_t W_t H_s W_s
        affinity: B affinity_dim H_s W_s
        """
        qkv = self.norm1(x)

        q, k = self.qk(qkv, affinity).chunk(2, dim=-1) # B (H_s W_s) C
        v = self.v(qkv) # B C H_t W_t H_s W_s

        q = rearrange(q, 'B S (C H) -> B S H C', H=self.nhead) + self.pos_embed
        k = rearrange(k, 'B S (C H) -> B S H C', H=self.nhead) + self.pos_embed
        H_s, W_s, H_t, W_t = v.shape[-4:]
        v = rearrange(v, 'B (H C) H_t W_t H_s W_s -> B (H_s W_s) H (C H_t W_t)', H=self.nhead)

        attn = rearrange(self.attn(q, k, v), 'B (H_s W_s) H (C H_t W_t) -> B (H C) H_t W_t H_s W_s', H_s=H_s, W_s=W_s, H_t=H_t, W_t=W_t)

        x = x + attn
        x = x + self.mlp(self.norm2(x))
        return x

# LinearConv4d definition (from transformer.py)
class LinearConv4d(nn.Module):
    def __init__(self, in_channel, out_channel, affinity_dim, target_proj_dim, input_corr_size=(8, 8, 8, 8),  kernel_size=(3, 3, 3, 3), stride=(2, 2, 2, 2), padding=(1, 1, 1, 1), group=1):
        super().__init__()

        self.conv = nn.Sequential(
            Conv4d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.GroupNorm(group, out_channel)
        )

        output_dim = out_channel * reduce(lambda x, y: x * y, [c // s for c, s in zip(input_corr_size[:2], stride[:2])])
        self.linear = nn.Linear(
            output_dim + affinity_dim,
            target_proj_dim
        )
    
    def forward(self, x, affinity):
        assert len(x.shape) == 6, 'input should be in shape B C H_t W_t H_s W_s'
        assert len(affinity.shape) == 4, 'affinity should be in shape B C H_s W_s'
        x = self.conv(x)
        x = torch.cat((rearrange(x, 'B C H_t W_t H_s W_s -> B (H_s W_s) (C H_t W_t)'), rearrange(affinity, 'B C H W -> B (H W) C')), dim=-1)
        x = self.linear(x)
        return x

# SelfAttention definition (from transformer.py)
class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        QK = torch.einsum("nlhd,nshd->nlsh", q, k)

        softmax_temp = 1. / q.size(3) ** .5
        attn = torch.softmax(softmax_temp * QK, dim=2)
        queried_values = torch.einsum("nlsh,nshd->nlhd", attn, v)

        return queried_values.contiguous()

# MLPConv4d definition (from transformer.py)
class MLPConv4d(nn.Module):
    def __init__(self, in_channel, mlp_ratio=4., kernel_size=(3, 3, 3, 3), stride=(2, 2, 2, 2), padding=(1, 1, 1, 1)):
        super().__init__()

        self.mlp = nn.Sequential(
            Conv4d(
                in_channels=in_channel,
                out_channels=int(in_channel * mlp_ratio),
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.GELU(),
            Conv4d(
                in_channels=int(in_channel * mlp_ratio),
                out_channels=in_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        )
    
    def forward(self, x):
        return self.mlp(x)

# unnormalise_and_convert_mapping_to_flow method definition (from mod.py)
def unnormalise_and_convert_mapping_to_flow(map):
    # here map is normalised to -1;1
    # we put it back to 0,W-1, then convert it to flow
    B, C, H, W = map.size()
    mapping = torch.zeros_like(map)
    # mesh grid
    mapping[:,0,:,:] = (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0 # unormalise
    mapping[:,1,:,:] = (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0 # unormalise

    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if mapping.is_cuda:
        grid = grid.cuda()
    flow = mapping - grid
    return flow





class CATs(nn.Module):
    def __init__(self, inch, affinity_dropout=.5, nhead=4):
        super(CATs, self).__init__()
        self.final_corr_size = 32

        self.early_conv = nn.ModuleList([
            Encoder4D( # Encoder for conv_5
                corr_levels=(inch[0], nhead),
                kernel_size=(
                    (3, 3, 3, 3),
                ),
                stride=(
                    (2, 2, 2, 2),
                ),
                padding=(
                    (1, 1, 1, 1),
                ),
                group=(1,),
            ),
            Encoder4D( # Encoder for conv_4
                corr_levels=(inch[1], nhead),
                kernel_size=(
                    (3, 3, 3, 3),
                ),
                stride=(
                    (2, 2, 2, 2),
                ),
                padding=(
                    (1, 1, 1, 1),
                ),
                group=(1,),
            ),
            Encoder4D( # Encoder for conv_3
                corr_levels=(inch[2], nhead),
                kernel_size=(
                    (3, 3, 3, 3),
                ),
                stride=(
                    (2, 2, 2, 2),
                ),
                padding=(
                    (1, 1, 1, 1),
                ),
                group=(1,),
            ),
        ])
        
        self.transformers = nn.ModuleList([
            Transformer(
                in_channel=nhead,
                depth=1,
                affinity_dim=128,
                target_proj_dim=512,
                nhead=nhead,
                mlp_ratio=4.,
                input_corr_size=(8, 8, 8, 8),
                stride=(1, 1, 1, 1),
                group=1,
            ),
            Transformer(
                in_channel=nhead,
                depth=1,
                affinity_dim=128,
                target_proj_dim=512,
                nhead=nhead,
                mlp_ratio=4.,
                input_corr_size=(16, 16, 16, 16),
                stride=(2, 2, 1, 1),
                group=1,
            ),
            Transformer(
                in_channel=nhead,
                depth=1,
                affinity_dim=128,
                target_proj_dim=512,
                nhead=nhead,
                mlp_ratio=4.,
                input_corr_size=(32, 32, 32, 32),
                kernel_size=(5, 5, 3, 3),
                stride=(4, 4, 1, 1),
                padding=(2, 2, 1, 1),
                group=1,
            ),
        ])

        self.affinity_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(True),
                nn.Dropout2d(affinity_dropout)
            ),
            nn.Sequential(
                nn.Conv2d(1024, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(True),
                nn.Dropout2d(affinity_dropout)
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(True),
                nn.Dropout2d(affinity_dropout)
            )
        ])

        self.upscale = nn.ModuleList([
            Interpolate4d(size=(16, 16, 16, 16)),
            Interpolate4d(size=(32, 32, 32, 32)),
        ])
    
        self.x_normal = np.linspace(-1,1,self.final_corr_size)
        self.x_normal = nn.Parameter(torch.tensor(self.x_normal, dtype=torch.float, requires_grad=False))
        self.y_normal = np.linspace(-1,1,self.final_corr_size)
        self.y_normal = nn.Parameter(torch.tensor(self.y_normal, dtype=torch.float, requires_grad=False))
        
    def softmax_with_temperature(self, x, beta, d = 1):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M # subtract maximum value for stability
        exp_x = torch.exp(x/beta)
        exp_x_sum = exp_x.sum(dim=d, keepdim=True)
        return exp_x / exp_x_sum

    def soft_argmax(self, corr, beta=0.02):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        b,_,h,w = corr.size()
        
        corr = self.softmax_with_temperature(corr, beta=beta, d=1)
        corr = corr.view(-1,h,w,h,w) # (target hxw) x (source hxw)

        grid_x = corr.sum(dim=1, keepdim=False) # marginalize to x-coord.
        x_normal = self.x_normal.expand(b,w)
        x_normal = x_normal.view(b,w,1,1)
        grid_x = (grid_x*x_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        
        grid_y = corr.sum(dim=2, keepdim=False) # marginalize to y-coord.
        y_normal = self.y_normal.expand(b,h)
        y_normal = y_normal.view(b,h,1,1)
        grid_y = (grid_y*y_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        return grid_x, grid_y

    def forward(self, hypercorr_pyramid, target_feats, source_feats):
        target_feats = [proj(x[:, -1]) for x, proj in zip(target_feats, self.affinity_proj)]
        source_feats = [proj(x[:, -1]) for x, proj in zip(source_feats, self.affinity_proj)]

        corr5 = self.early_conv[0](hypercorr_pyramid[0])
        corr4 = self.early_conv[1](hypercorr_pyramid[1])
        corr3 = self.early_conv[2](hypercorr_pyramid[2])

        corr5 = corr5 + self.transformers[0](corr5, source_feats[0]) + transpose4d(self.transformers[0](transpose4d(corr5), target_feats[0]))
        corr4 = corr4 + self.upscale[0](corr5)
        corr4 = corr4 + self.transformers[1](corr4, source_feats[1]) + transpose4d(self.transformers[1](transpose4d(corr4), target_feats[1]))
        corr3 = corr3 + self.upscale[1](corr4)
        corr3 = corr3 + self.transformers[2](corr3, source_feats[2]) + transpose4d(self.transformers[2](transpose4d(corr3), target_feats[2]))
        corr3 = corr3.mean(1)

        grid_x, grid_y = self.soft_argmax(rearrange(corr3, 'B H_t W_t H_s W_s -> B (H_s W_s) H_t W_t'))

        grid = torch.cat((grid_x, grid_y), dim=1)
        flow = unnormalise_and_convert_mapping_to_flow(grid)

        return flow