from .layers import ResnetBlockDDPM, AttnBlock, Upsample, Downsample, ddpm_conv3x3
from .input_embedding import InputProcessingImage

import torch.nn as nn
import torch
import functools

class ARM_UNet(nn.Module):
    def __init__(self, binary, num_classes, ch, out_ch, input_channels, ch_mult, num_res_blocks, full_attn_resolutions, num_heads, dropout, max_time=3072., weave_attn=False):
        super().__init__()
        self.binary = binary
        self.num_classes = num_classes
        self.ch = ch
        self.out_ch = out_ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.full_attn_resolutions = full_attn_resolutions
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_time = max_time
        self.weave_attn = weave_attn

        self.act = nn.SiLU()
        self.num_resolutions = len(ch_mult)
        nf = self.ch     # should be 256 (or 2 for binary)
        ResnetBlock = functools.partial(ResnetBlockDDPM, temb_dim=4 * nf, dropout=self.dropout)
        modules = []


        modules.append(InputProcessingImage(
            binary = self.binary,
            num_classes=self.num_classes, 
            num_channels=self.ch * self.ch_mult[0], 
            input_channels=input_channels,
            max_time=self.max_time,
        ))

        # Downsampling block
        #modules.append(ddpm_conv3x3(channels, nf))
        hs_c = [nf]
        in_ch = nf
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                #if out_ch in self.full_attn_resolutions:
                if self.weave_attn:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)
            if i_level != self.num_resolutions - 1:
                modules.append(Downsample(channels=in_ch, with_conv=True))
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=in_ch))

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch
                #if out_ch in self.full_attn_resolutions:
                if self.weave_attn:
                    modules.append(AttnBlock(channels=in_ch))
            if i_level != 0:
                modules.append(Upsample(channels=in_ch, with_conv=True))

        assert(not hs_c)
        modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
        modules.append(ddpm_conv3x3(in_ch, self.out_ch, init_scale=0.))
        self.all_modules = nn.ModuleList(modules)

        # print(self)

    def forward(self, x, t, mask):
        modules = self.all_modules
        m_idx = 0

        B, C, H, W = x.shape
        assert(H == W)
        assert(t.shape == (B,))

        h_first, temb = modules[m_idx](x, t, mask)
        m_idx += 1

        # We don't want to access x, t or mask directly, but only their embeddings
        # via h_first and temb.
        del x, t, mask
        assert(h_first.dtype in (torch.float32, torch.float64))

        num_resolutions = len(self.ch_mult)
        ch = self.ch
        y = None

        # Downsampling
        hs = [h_first]
        in_ch = ch
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb=temb)
                m_idx += 1
                #if h.shape[1] in self.full_attn_resolutions:
                if self.weave_attn:
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)
            # Downsample
            if i_level != self.num_resolutions - 1:
                hs.append(modules[m_idx](hs[-1]))
                m_idx += 1

        # Middle
        h = hs[-1]
        h = modules[m_idx](h, temb=temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb=temb)
        m_idx += 1

        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb=temb)
                m_idx += 1
                #if h.shape[1] in self.full_attn_resolutions:
                if self.weave_attn:
                    h = modules[m_idx](h)
                    m_idx += 1
            # Upsample
            if i_level != 0:
                h = modules[m_idx](h)
                m_idx += 1

        # End.
        assert(not hs)
        h = self.act(modules[m_idx](h))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        assert(m_idx == len(modules))

        return h



