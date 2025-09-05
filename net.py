import torch
import torch.nn as nn
from collections import OrderedDict


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=padding,
        bias=True,
        dilation=dilation,
        groups=groups,
    )


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == "batch":
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == "instance":
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError(
            "normalization layer [{:s}] is not found".format(norm_type)
        )
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == "reflect":
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == "replicate":
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError(
            "padding layer [{:s}] is not implemented".format(pad_type)
        )
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(
    in_nc,
    out_nc,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    pad_type="zero",
    norm_type=None,
    act_type="relu",
):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != "zero" else None
    padding = padding if pad_type == "zero" else 0

    c = nn.Conv2d(
        in_nc,
        out_nc,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        groups=groups,
    )
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == "relu":
        layer = nn.ReLU(inplace)
    elif act_type == "lrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            "activation layer [{:s}] is not found".format(act_type)
        )
    return layer


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output


def mean_channels(F):
    assert F.dim() == 4
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert F.dim() == 4
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (
        F.size(2) * F.size(3)
    )
    return F_variance.pow(0.5)


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(
    in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1
):
    conv = conv_layer(
        in_channels, out_channels * (upscale_factor**2), kernel_size, stride
    )
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class HDB(nn.Module):
    """
    Input: U in R^{F x H x W}
    Output: V in R^{F x H x W}
    Internals:
      - local 1x1 mixing (U_m)
      - three depthwise branches (DW 3x3 -> PW proj to s_i)
      - remainder 1x1 to keep F - S_tot channels
      - concat -> 1x1 fuse -> residual add
    """

    def __init__(self, in_channels, s_splits=(20, 12, 8), act_type="lrelu"):
        super(HDB, self).__init__()
        F_ch = in_channels
        s1, s2, s3 = s_splits
        S_tot = s1 + s2 + s3
        assert S_tot <= F_ch, "Sum of splits must be <= in_channels"
        rem_ch = F_ch - S_tot

        self.F = F_ch
        self.splits = s_splits

        # local mixing (1x1)
        self.mix = conv_layer(F_ch, F_ch, kernel_size=1)

        # depthwise convs (3 parallel branches)
        # depthwise conv: groups=F
        self.dw1 = conv_layer(F_ch, F_ch, kernel_size=3, groups=F_ch)
        self.pw1 = conv_layer(F_ch, s1, kernel_size=1)

        self.dw2 = conv_layer(F_ch, F_ch, kernel_size=3, groups=F_ch)
        self.pw2 = conv_layer(F_ch, s2, kernel_size=1)

        self.dw3 = conv_layer(F_ch, F_ch, kernel_size=3, groups=F_ch)
        self.pw3 = conv_layer(F_ch, s3, kernel_size=1)

        # remainder stream projection (1x1)
        self.rem = conv_layer(F_ch, rem_ch, kernel_size=1)

        # fuse back to F channels
        self.fuse = conv_layer(F_ch, F_ch, kernel_size=1)

        # activation
        self.act = activation(act_type, neg_slope=0.05)

    def forward(self, x):
        # x : (B, F, H, W)
        U_m = self.mix(x)  # (B, F, H, W)
        U_m_act = self.act(U_m)

        # branch 1
        b1 = self.dw1(U_m_act)  # (B, F, H, W)
        b1 = self.act(b1)
        d1 = self.pw1(b1)  # (B, s1, H, W)

        # branch 2
        b2 = self.dw2(U_m_act)
        b2 = self.act(b2)
        d2 = self.pw2(b2)  # (B, s2, H, W)

        # branch 3
        b3 = self.dw3(U_m_act)
        b3 = self.act(b3)
        d3 = self.pw3(b3)  # (B, s3, H, W)

        # remainder
        r = self.rem(U_m_act)  # (B, F-S_tot, H, W)

        # concat in channel dim -> total F channels
        out = torch.cat([d1, d2, d3, r], dim=1)  # (B, F, H, W)

        out = self.act(out)
        out = self.fuse(out)  # (B, F, H, W)

        # residual add (shallow)
        out = out + x
        return out


class SAG(nn.Module):
    """
    Global squeeze -> small bottleneck MLP -> sigmoid gates applied channelwise.
    Input/Output: (B, F, H, W) -> (B, F, H, W)
    """

    def __init__(self, channels, hidden=8):
        super(SAG, self).__init__()
        self.channels = channels
        self.pool = nn.AdaptiveAvgPool2d(1)  # -> (B, C, 1, 1)
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, channels, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, _, _ = x.size()
        s = self.pool(x).view(b, c)  # (B, C)
        s = self.fc1(s)  # (B, hidden)
        s = self.relu(s)
        s = self.fc2(s)  # (B, C)
        s = self.sig(s).view(b, c, 1, 1)  # (B, C, 1, 1)
        return x * s  # broadcast multiply


class HBDN(nn.Module):
    """
    Hierarchical Bottleneck Distillation Network (HBDN)
    - head: 3x3 conv -> F channels
    - body: N x HDB blocks (each returns F channels)
    - aggregation: concat all block outputs (N*F) -> 1x1 proj -> 3x3 smooth
    - add head residual -> SAG -> recon conv -> pixelshuffle
    """

    def __init__(
        self,
        in_nc=3,
        nf=80,
        num_modules=12,
        out_nc=3,
        upscale=4,
        s_splits=[20, 12, 8],
        sag_hidden=8,
        act_type="lrelu",
    ):
        super(HBDN, self).__init__()
        self.in_nc = in_nc
        self.nf = nf
        self.N = num_modules
        self.upscale = upscale

        # head
        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)

        # body: N HDBs
        self.body = nn.ModuleList(
            [HDB(nf, s_splits=s_splits, act_type=act_type) for _ in range(self.N)]
        )

        # aggregation: concat N*F -> proj to F -> smooth 3x3
        self.agg_proj = conv_layer(nf * self.N, nf, kernel_size=1)
        self.smooth = conv_layer(nf, nf, kernel_size=3)

        # SAG gate
        self.sag = SAG(nf, hidden=sag_hidden)

        # reconstruction: conv to C_out * S^2 then pixelshuffle
        out_ch = out_nc * (upscale**2)
        self.recon = conv_layer(nf, out_ch, kernel_size=3)
        self.pixel_shuffle = nn.PixelShuffle(upscale)

        # activation
        self.act = activation(act_type, neg_slope=0.05)

    def forward(self, x):
        # x: (B, 3, H, W)
        head = self.fea_conv(x)  # (B, F, H, W)

        feats = []
        inp = head
        for i, block in enumerate(self.body):
            out = block(inp)  # (B, F, H, W)
            feats.append(out)
            inp = out  # feed-forward chain (as in RFDN/IMDN)

        # aggregate all intermediate block outputs
        cat = torch.cat(feats, dim=1)  # (B, N*F, H, W)
        proj = self.agg_proj(cat)  # (B, F, H, W)
        smooth = self.smooth(self.act(proj))

        # global residual add
        fused = smooth + head  # (B, F, H, W)

        # SAG gating
        gated = self.sag(fused)  # (B, F, H, W)

        # reconstruction + pixelshuffle
        pre = self.recon(gated)  # (B, out_nc * S^2, H, W)
        out = self.pixel_shuffle(pre)  # (B, out_nc, S*H, S*W)
        return out

    def extra_repr(self):
        return f"nf={self.nf}, N={self.N}, splits={self.body[0].splits if len(self.body) > 0 else None}, upscale={self.upscale}"


if __name__ == "__main__":
    model = HBDN(
        in_nc=3,
        nf=80,
        num_modules=12,
        out_nc=3,
        upscale=4,
        s_splits=(20, 12, 8),
        sag_hidden=8,
    )
    print(model)
    input = torch.randn(1, 3, 64, 64)
    output = model(input)
    print(output.shape)
