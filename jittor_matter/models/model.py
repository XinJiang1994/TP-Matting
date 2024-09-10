import jittor as jt
import jittor.nn as nn
from .utils import erode


class TgtFilterMatterWrapper(nn.Module):
    def __init__(self, tgt_filter, matting_model, r=0.01):
        super().__init__()
        self.tgt_filter = tgt_filter
        self.matting_model = matting_model
        self.r = r
        self.isize = 512
        # self.dilate=Dilation2d(1, 3, 5, soft_max=False)
        # self.erode=Erosion2d(1, 3, 5, soft_max=False)
        self.cached_pha = []
        self.tm_mask = None
        self.counter = 0
        self.buffer_size = 1

    def update_tm_mask(self, pha):
        self.cached_pha.append(pha)
        if len(self.cached_pha) > self.buffer_size:
            self.cached_pha.pop(0)
            self.tm_mask = jt.mean(self.cached_pha, dim=0)

    def init_buffer(self):
        self.tgt_filter.init_x_t()
        self.cached_pha = []
        self.tm_mask = None
        self.counter = 0

    def execute(self, x, x_tgt, inference=False, tm=False):
        # x,x_tgt = xxt.split([3, 3], dim=1)
        downsample_ratio = self.isize/max(x.shape[-2:])
        pha_sm, fgr_base, pha_base = self.tgt_filter(x, x_tgt, inference)
        mask = nn.interpolate(pha_sm, size=x.shape[-2:], mode="bilinear")
        if inference and tm and self.tm_mask is not None:
            mask = 0.6*mask + 0.4*self.tm_mask
        x_c4 = jt.cat((x, mask), dim=1)
        fgr, pha, details = self.matting_model(
            x_c4, x, downsample_ratio=downsample_ratio)
        if inference and tm and self.counter % self.buffer_size == 0:
            self.update_tm_mask(pha)
        self.counter += 1
        return pha, fgr, details, pha_sm, fgr_base, pha_base
