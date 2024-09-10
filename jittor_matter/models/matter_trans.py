import jittor as jt
import jittor.nn as nn
from .sr_module import SRNet

from .utils import padding, unpadding


class TgtFilter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
        isize_vit=256,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder
        self.sr = SRNet(n_cls)
        self.isize_vit = isize_vit
        self.buffer_size = 2
        self.buffer_update_seq = 15
        self.buffer_timer = 0
        self.buffer = []
        self.x_t = None

    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def init_x_t(self):
        self.x_t = None

    def execute(self, im, tgt, inference=False):
        # im_src=im
        # H_src, W_src = im_src.size(2), im_src.size(3)
        im = jt.nn.interpolate(
            im, size=(self.isize_vit, self.isize_vit), mode="bilinear")
        tgt = jt.nn.interpolate(tgt, size=(
            self.isize_vit, self.isize_vit), mode="bilinear")

        if not inference:
            tgt = padding(tgt, self.patch_size)
            x_t = self.encoder(tgt, return_features=True)
        else:
            if self.x_t is None:
                tgt = padding(tgt, self.patch_size)
                x_t = self.encoder(tgt, return_features=True)
                self.x_t = x_t
            else:
                x_t = self.x_t
                # x_t = jt.zeros_like(self.x_t)

        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]
        x_t = x_t[:, num_extra_tokens:]

        tokens, _ = self.decoder(x, x_t, (H, W))  # N x 4 x 32 x 32
        # if inference and self.buffer_timer % self.buffer_update_seq == 0:
        #     self.update_x_t(enc_cross, num_extra_tokens)

        tokens = jt.nn.interpolate(tokens, size=(H, W), mode="bilinear")
        tokens = unpadding(tokens, (H_ori, W_ori))
        out = self.sr(im, tokens)
        pha = jt.sigmoid(out)
        named_tokens = tokens[:, :4, :, :]
        pha_base, fgr_base = jt.sigmoid(named_tokens).split([1, 3], dim=1)
        return pha, fgr_base, pha_base

    def update_x_t(self, enc_cross, num_extra_tokens):
        self.buffer.append(enc_cross)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        self.x_t[:, num_extra_tokens:] = jt.mean(jt.concat(
            [self.x_t[:, num_extra_tokens:]]+self.buffer, dim=0), dim=0, keepdims=True)
        self.buffer_timer = 0

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)
