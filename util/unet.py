import math
import torch
from torch import nn
from einops import rearrange
from inspect import isfunction
from util.new_ffc import FFC_BN_ACT, TransFFC, FFC

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        if type(x) is tuple:
            return (self.fn(x, *args, **kwargs)[0] + x[0], self.fn(x, *args, **kwargs)[1] + x[1])
        else:
            return self.fn(x, *args, **kwargs) + x
class NormalResidual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)
    #return Trans_FFC_BN_ACT(dim, dim, 4, stride=2, padding=1, ratio_gin=0, ratio_gout=0)
    
def FFCUpsample(dim, ratio_gin=0, ratio_gout=0):
    return TransFFC(dim, dim, 4, stride=2, padding=1, ratio_gin=ratio_gin, ratio_gout=ratio_gout, upsample=True)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)
    #return FFC_BN_ACT(dim, dim, 4, stride=2, padding=1, ratio_gin=0, ratio_gout=ratio_gout)
    
def FFCDownsample(dim, ratio_gout=0):
    return FFC_BN_ACT(dim, dim, 4, stride=2, padding=1, ratio_gin=0, ratio_gout=ratio_gout)

def Splitsample(dim, ratio_gout=0):
    #return nn.Conv2d(dim, dim, 4, 2, 1)
    return FFC_BN_ACT(dim, dim, 3, stride=1, padding=1, ratio_gin=0, ratio_gout=ratio_gout)

def SkipSplit(dim, ratio_gout=0):
    #return nn.Conv2d(dim, dim, 4, 2, 1)
    return FFC_BN_ACT(dim, dim, 3, stride=1, padding=1, ratio_gin=0, ratio_gout=ratio_gout)

class LayerNorm(nn.Module):
    def __init__(self, dim, split_ratio = 0, eps = 1e-5):
        super().__init__()
        self.eps = eps
        
        self.split_num = int(dim * split_ratio)
        self.remaining_num = dim - self.split_num
        
        # Define parameters based on the split
        self.g = nn.Parameter(torch.ones(1, self.remaining_num, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, self.remaining_num, 1, 1))
        
        if self.split_num > 0:
            self.g2 = nn.Parameter(torch.ones(1, self.split_num, 1, 1))
            self.b2 = nn.Parameter(torch.zeros(1, self.split_num, 1, 1))
        else:
            self.g2 = None
            self.b2 = None

    def forward(self, x):
        if type(x) is tuple:
            x1,x2 = x
            var = torch.var(x1, dim = 1, unbiased = False, keepdim = True)
            mean = torch.mean(x1, dim = 1, keepdim = True)
            y = (x1 - mean) / (var + self.eps).sqrt() * self.g + self.b
            if type(x2) is not int:
                var2 = torch.var(x2, dim = 1, unbiased = False, keepdim = True)
                mean2 = torch.mean(x2, dim = 1, keepdim = True)
                y2 = (x2 - mean2) / (var2 + self.eps).sqrt() * self.g2 + self.b2
                return y, y2
            return y,x2
        else:
            var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
            mean = torch.mean(x, dim = 1, keepdim = True)
            return (x - mean) / (var + self.eps).sqrt() * self.g + self.b
class NormalLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b
class PreNorm(nn.Module):
    def __init__(self, dim, fn, split_ratio=0):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim, split_ratio=split_ratio)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
    
class NormalPreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# building block modules
class Custom_GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = nn.GELU()

    def forward(self, x):
        x1,x2 = x
        x1 = self.gelu(x1)
        if torch.is_tensor(x2):
            x2 = self.gelu(x2)
        x = x1,x2
        return x
class ConvNextBlock(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, *, time_emb_dim = None, mult = 2, norm = True, ratio_gin=0, ratio_gout=0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, dim*2)
        ) if exists(time_emb_dim) else None

        self.ds_conv = FFC_BN_ACT(dim, dim, 7, padding= 3, groups = dim, ratio_gin=ratio_gin, ratio_gout=ratio_gout)
        #self.ds_conv = nn.Conv2d(dim, dim, 7, padding = 3, groups = dim)

        self.net = nn.Sequential(
            LayerNorm(dim, split_ratio=ratio_gout) if norm else nn.Identity(),
            #nn.Conv2d(dim, dim_out * mult, 3, padding = 1),
            FFC_BN_ACT(dim, dim_out * mult, 3, padding = 1, ratio_gin=ratio_gin, ratio_gout=ratio_gout),
            Custom_GELU(),
            #nn.Conv2d(dim_out * mult, dim_out, 3, padding = 1)
            FFC_BN_ACT(dim_out * mult, dim_out, 3, padding = 1, ratio_gin=ratio_gin, ratio_gout=ratio_gout),
        )

        #self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.res_conv = FFC_BN_ACT(dim, dim_out, 1, ratio_gin=ratio_gin, ratio_gout=ratio_gout) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        h = self.ds_conv(x)
        h1,h2 = h
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            condition = self.mlp(time_emb)
            condition = rearrange(condition, 'b c -> b c 1 1')
            if type(x)is tuple:
                if type(x[1]) is not int:
                    channels = condition.shape[1]
                    global_in_num = int(channels * 0.75)
                    condition, condition2 = condition[:, :-global_in_num], condition[:, -global_in_num:]
                weight, bias = torch.split(condition, x[0].shape[1],dim=1) #### x.shape[1] is channel
                if type(x[1]) is not int:
                    weight2, bias2 = torch.split(condition2, x[1].shape[1],dim=1) #### x.shape[1] is channel
            else:
                weight, bias = torch.split(condition, x.shape[1],dim=1) #### x.shape[1] is channel
            if type(h2) is int:
                h1 = h1 * (1 + weight) + bias
            else:
                h1 = h1 * (1 + weight) + bias
                h2 = h2 * (1 + weight2) + bias2
        h = h1,h2
        h = self.net(h)
        return (h[0] + self.res_conv(x)[0], h[1] + self.res_conv(x)[1])
    
class FinalNextBlock(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, *, time_emb_dim = None, mult = 2, norm = True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, dim*2)
        ) if exists(time_emb_dim) else None

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding = 3, groups = dim)

        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding = 1)
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            condition = self.mlp(time_emb)
            condition = rearrange(condition, 'b c -> b c 1 1')
            weight, bias = torch.split(condition, x.shape[1],dim=1)
            h = h * (1 + weight) + bias

        h = self.net(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, split_ratio=0):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.split_num = int(dim * split_ratio)
        self.remaining_num = dim - self.split_num
        
        # Define parameters based on the split
        self.to_qkv = nn.Conv2d(self.remaining_num, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, self.remaining_num, 1)
        
        if self.split_num > 0:
            self.to_qkv2 = nn.Conv2d(self.split_num, hidden_dim * 3, 1, bias = False)
            self.to_out2 = nn.Conv2d(hidden_dim, self.split_num, 1)
        else:
            self.to_qkv2 = None
            self.to_out2 = None

    def forward(self, x):
        if type(x) is tuple:
            b, c, h, w = x[0].shape
            qkv = self.to_qkv(x[0]).chunk(3, dim = 1)
            q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
            q = q * self.scale

            k = k.softmax(dim = -1)
            context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

            out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
            out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
            if type(x[1]) is not int:
                b2, c2, h2, w2 = x[1].shape
                qkv2 = self.to_qkv2(x[1]).chunk(3, dim = 1)
                q2, k2, v2 = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv2)
                q2 = q2 * self.scale

                k2 = k2.softmax(dim = -1)
                context2 = torch.einsum('b h d n, b h e n -> b h d e', k2, v2)

                out2 = torch.einsum('b h d e, b h d n -> b h e n', context2, q2)
                out2 = rearrange(out2, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h2, y = w2)
                return self.to_out(out), self.to_out2(out2)
            return self.to_out(out), x[1]
        else:
            b, c, h, w = x.shape
            qkv = self.to_qkv(x).chunk(3, dim = 1)
            q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
            q = q * self.scale

            k = k.softmax(dim = -1)
            context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

            out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
            out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
            return self.to_out(out)
class NormalLinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        q = q * self.scale
        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)
class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)

# model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        with_time_emb = True,
        residual = False,
        attn_skip = False,
    ):
        super().__init__()
        self.channels = channels
        self.residual = residual
        print("Is Time embed used ? ", with_time_emb)
        if isinstance(dim_mults, list):
            dim_mults = tuple(dim_mults)
        assert isinstance(dim_mults, tuple), "Error: dim_mults is not a tuple"
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.model_depth = len(dim_mults)

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                FinalNextBlock(dim_in, dim_out, time_emb_dim = time_dim, norm = ind != 0),
                FinalNextBlock(dim_out, dim_out, time_emb_dim = time_dim),
                NormalResidual(NormalPreNorm(dim_out, NormalLinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else Splitsample(dim_out, ratio_gout=0.75)
            ]))


        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim, ratio_gin=0.75, ratio_gout=0.75)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim, split_ratio=0.75), split_ratio=0.75))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim, ratio_gin=0.75, ratio_gout=0.75)
        self.concat = ConcatTupleLayer()

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                FinalNextBlock(dim_out * 2, dim_in, time_emb_dim = time_dim),
                FinalNextBlock(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            FinalNextBlock(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        orig_x = x
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        for convnext, convnext2, attn, downsample in self.downs:
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        x = self.concat(x)
        # for idx, item in enumerate(h):
        #     h[idx] = self.concat(item)

        for convnext, convnext2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            x = upsample(x)
        if self.residual:
            return self.final_conv(x) + orig_x
        return self.final_conv(x)
        ### Maybe add tanh to limit the range to -1 to 1
        
class UnetFFC(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        with_time_emb = True,
        residual = False,
        num=4,
    ):
        super().__init__()
        self.channels = channels
        self.residual = residual
        print("Is Time embed used ? ", with_time_emb)
        if isinstance(dim_mults, list):
            dim_mults = tuple(dim_mults)
        assert isinstance(dim_mults, tuple), "Error: dim_mults is not a tuple"
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.model_depth = len(dim_mults)

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.mids = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                FinalNextBlock(dim_in, dim_out, time_emb_dim = time_dim, norm = ind != 0),
                FinalNextBlock(dim_out, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else Splitsample(dim_out, ratio_gout=0.75)
            ]))


        mid_dim = dims[-1]
        for i in range(num):
            self.mids.append(nn.ModuleList([
                ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim, ratio_gin=0.75, ratio_gout=0.75),
                Residual(PreNorm(mid_dim, LinearAttention(mid_dim, split_ratio=0.75), split_ratio=0.75)),
                ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim, ratio_gin=0.75, ratio_gout=0.75),
            ]))

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                FinalNextBlock(dim_out * 2, dim_in, time_emb_dim = time_dim),
                FinalNextBlock(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            FinalNextBlock(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )
        
        self.concat = ConcatTupleLayer()

    def forward(self, x, time):
        orig_x = x
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        for convnext, convnext2, attn, downsample in self.downs:
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        for convnext, attn, convnext2 in self.mids:
            # h1, h2 = x
            x = convnext(x, t)
            x = attn(x)
            x = convnext2(x, t)
            # x = x[0] + h1, x[1] + h2
        x = self.concat(x)
        # for idx, item in enumerate(h):
        #     h[idx] = self.concat(item)

        for convnext, convnext2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            x = upsample(x)
            
        if self.residual:
            return self.final_conv(x) + orig_x
        return self.final_conv(x)
        
class UnetAttention(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        with_time_emb = True,
        residual = False,
        attn_skip = False,
    ):
        super().__init__()
        self.channels = channels
        self.residual = residual
        print("Is Time embed used ? ", with_time_emb)
        if isinstance(dim_mults, list):
            dim_mults = tuple(dim_mults)
        assert isinstance(dim_mults, tuple), "Error: dim_mults is not a tuple"
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.model_depth = len(dim_mults)

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                FinalNextBlock(dim_in, dim_out, time_emb_dim = time_dim, norm = ind != 0),
                FinalNextBlock(dim_out, dim_out, time_emb_dim = time_dim),
                NormalResidual(NormalPreNorm(dim_out, NormalLinearAttention(dim_out))),
                NormalResidual(NormalPreNorm(dim_out, NormalLinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else Splitsample(dim_out, ratio_gout=0.75)
            ]))


        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim, ratio_gin=0.75, ratio_gout=0.75)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim, split_ratio=0.75), split_ratio=0.75))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim, ratio_gin=0.75, ratio_gout=0.75)
        self.concat = ConcatTupleLayer()

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                FinalNextBlock(dim_out * 2, dim_in, time_emb_dim = time_dim),
                FinalNextBlock(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            FinalNextBlock(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        orig_x = x
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        for convnext, convnext2, attn, skip_attn, downsample in self.downs:
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            assert not torch.isnan(x).any(), f"{x} NaN before skip attn"
            h.append(skip_attn(x))
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        x = self.concat(x)
        # for idx, item in enumerate(h):
        #     h[idx] = self.concat(item)

        for convnext, convnext2, attn, upsample in self.ups:
            a = h.pop()
            assert not torch.isnan(a).any(), f"{a} NaN before skip attn"
            x = torch.cat((x, a), dim=1)
            assert not torch.isnan(x).any(), f"{x} NaN after concat"
            x = convnext(x, t)
            assert not torch.isnan(x).any(), f"{x} NaN after convnext1"
            x = convnext2(x, t)
            assert not torch.isnan(x).any(), f"{x} NaN after convnext2"
            x = attn(x)
            assert not torch.isnan(x).any(), f"{x} NaN after attn"
            x = upsample(x)
            assert not torch.isnan(x).any(), f"{x} NaN after upsample"
        if self.residual:
            return self.final_conv(x) + orig_x
        assert not torch.isnan(x).any(), f"{x} NaN before final conv"
        return self.final_conv(x)       
        
if __name__ == "__main__":
    # from torchinfo import summary
    device="cuda:0"
    model = UnetFFC(
        dim=64,                     # 16 ==> 128
        dim_mults=(1, 2, 4),     # 16 ==> 128
        channels=3,
        residual=False,
        num=18,
    ).to(device)
    print(sum(p.numel() for p in model.parameters()))
    batch_size = 1
    input_size = (batch_size, 3, 256, 256)
    timestep_size = (batch_size,1)
    input = torch.randn(input_size).to(device)
    t = torch.randint(1, 21, (batch_size,), device=device).long()
    out = model(input, t)
    print(out.shape)
    # summary(model, input_data=(input,t))