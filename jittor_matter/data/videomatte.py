import os
import random
from jittor.dataset.dataset import Dataset
from PIL import Image
import numpy as np

from jittor_matter.data.utils import erode, random_erase


class VideoMatteDataset(Dataset):
    def __init__(self,
                 videomatte_dir,
                 background_image_dir,
                 background_video_dir,
                 size,
                 transform=None,
                 multi=True,
                 split='train',
                 dataset_name='videomatte'
                 ):
        super().__init__()
        self.split = split
        self.dataset_name = dataset_name
        self.background_image_dir = background_image_dir
        self.background_image_files = [x for x in os.listdir(
            background_image_dir) if 'checkpoint' not in x][:10000]
        self.background_video_dir = background_video_dir
        self.background_video_clips = sorted(os.listdir(background_video_dir))
        self.background_video_frames = [sorted([x for x in os.listdir(os.path.join(background_video_dir, clip)) if x.endswith('.jpg')])
                                        for clip in self.background_video_clips]

        self.videomatte_dir = videomatte_dir
        self.videomatte_clips = sorted(os.listdir(
            os.path.join(videomatte_dir, 'fgr_com')))
        print(self.videomatte_clips)
        # self.videomatte_clips = [x for x in self.videomatte_clips if x not in [
        #         '0073', '0139', '0151', '0173', '0454']]
        if split == 'val':
            self.videomatte_clips = [x for x in self.videomatte_clips if x in [
                '0073', '0139', '0151', '0173', '0454']]  # val data
        self.videomatte_frames = [sorted([x for x in os.listdir(os.path.join(videomatte_dir, 'fgr_com', clip)) if x.endswith('.jpg')])
                                  for clip in self.videomatte_clips]
        self.videomatte_idx = [(clip_idx, frame_idx)
                               for clip_idx in range(len(self.videomatte_clips))
                               for frame_idx in range(0, len(self.videomatte_frames[clip_idx]), 10)]
        self.multi = multi
        self.size = size
        self.transform = transform
        self.n_cls = 1+3  # fgr_pha, fgr
        self.set_attrs(total_len=len(self.videomatte_idx))

    def __getitem__(self, idx):
        if random.random() < 0.5:
            bgrs = self._get_random_image_background()
        else:
            bgrs = self._get_random_video_background()

        fgr, fgr_com, tgt, phas, pha_com, trimap = self._get_videomatte(idx)

        fgr, fgr_com, phas, pha_com, bgrs, trimap = self.transform(
            fgr, fgr_com, phas, pha_com, bgrs, trimap)
        # because it's a pair transform, so it returns (tgt,)
        tgt = self.transform(tgt)[0]
        if self.dataset_name == 'BGM':
            im = fgr_com
        elif self.multi:
            if random.random() < 0.8:
                if random.random() < 0.5:
                    im = pha_com*fgr_com+(1-pha_com)*bgrs
                else:
                    im = pha_com*fgr_com
            else:
                im = phas*fgr+(1-phas)*bgrs
        else:
            im = phas*fgr+(1-phas)*bgrs

        out = dict(ims=im)
        out['fgrs'] = fgr
        out['tgts'] = tgt
        out['phas'] = phas
        out['trimap'] = trimap

        return out

    def _get_random_image_background(self):
        with Image.open(os.path.join(self.background_image_dir, random.choice(self.background_image_files))) as bgr:
            bgr = self._downsample_if_needed(bgr.convert('RGB'))
        return bgr

    def _get_random_video_background(self):
        clip_idx = random.choice(range(len(self.background_video_clips)))
        frame_count = len(self.background_video_frames[clip_idx])
        frame_idx = random.choice(range(max(1, frame_count)))
        clip = self.background_video_clips[clip_idx]
        frame = self.background_video_frames[clip_idx][frame_idx]
        with Image.open(os.path.join(self.background_video_dir, clip, frame)) as bgr:
            bgr = self._downsample_if_needed(bgr.convert('RGB'))
        return bgr

    def _get_videomatte(self, idx):
        clip_idx, frame_idx = self.videomatte_idx[idx]
        clip = self.videomatte_clips[clip_idx]
        frame_count = len(self.videomatte_frames[clip_idx])
        frame_idx = (frame_idx + random.randint(0, 10)) % frame_count
        frame_tgt_idx = random.choice(range(max(1, frame_count)))
        frame = self.videomatte_frames[clip_idx][frame_idx]
        frame_tgt = self.videomatte_frames[clip_idx][frame_tgt_idx]
        with Image.open(os.path.join(self.videomatte_dir, 'fgr', clip, frame)) as fgr, \
                Image.open(os.path.join(self.videomatte_dir, 'fgr', clip, frame_tgt)) as tgt, \
                Image.open(os.path.join(self.videomatte_dir, 'fgr_com', clip, frame)) as fgr_com, \
                Image.open(os.path.join(self.videomatte_dir, 'pha', clip, frame)) as pha, \
                Image.open(os.path.join(self.videomatte_dir, 'pha_com', clip, frame)) as pha_com:
            if self.split == 'train':
                trimap = Image.open(os.path.join(
                    self.videomatte_dir, 'trimap', clip, frame))
            else:
                trimap = pha
            fgr = self._downsample_if_needed(fgr.convert('RGB'))
            try:
                tgt = self._downsample_if_needed(tgt.convert('RGB'))
            except:
                print('######## Frame have problems: ', os.path.join(
                    self.videomatte_dir, 'fgr', clip, frame_tgt))
                exit(200)
            fgr_com = self._downsample_if_needed(fgr_com.convert('RGB'))
            pha = self._downsample_if_needed(pha.convert('L'))
            pha_com = self._downsample_if_needed(pha_com.convert('L'))
            trimap = self._downsample_if_needed(trimap.convert('L'))
        return fgr, fgr_com, tgt, pha, pha_com, trimap

    def _downsample_if_needed(self, img):
        w, h = img.size
        if min(w, h) > self.size:
            scale = self.size / min(w, h)
            w = int(scale * w)
            h = int(scale * h)
            img = img.resize((w, h))
        return img

    @property
    def unwrapped(self):
        return self


class VideoMatteDatasetSim(Dataset):
    def __init__(self,
                 videomatte_dir,
                 background_image_dir,
                 background_video_dir,
                 size,
                 transform=None,
                 multi=True,
                 split='train',
                 dataset_name='videomatte'
                 ):
        super().__init__()
        self.split = split
        self.dataset_name = dataset_name
        self.background_image_dir = background_image_dir
        self.background_image_files = [x for x in os.listdir(
            background_image_dir) if 'checkpoint' not in x][:10000]
        self.background_video_dir = background_video_dir
        self.background_video_clips = sorted(os.listdir(background_video_dir))
        self.background_video_frames = [sorted([x for x in os.listdir(os.path.join(background_video_dir, clip)) if x.endswith('.jpg')])
                                        for clip in self.background_video_clips]

        self.videomatte_dir = videomatte_dir
        self.videomatte_clips = sorted(os.listdir(
            os.path.join(videomatte_dir, 'fgr_com')))
        self.videomatte_clips = [x for x in self.videomatte_clips if x not in [
            '0073', '0139', '0151', '0173', '0454']]  # val data
        if split=='val':
            self.videomatte_clips=['0073', '0139', '0151', '0173', '0454']

        self.videomatte_frames = [sorted([x for x in os.listdir(os.path.join(videomatte_dir, 'fgr_com', clip)) if x.endswith('.jpg')])
                                  for clip in self.videomatte_clips]
        self.videomatte_idx = [(clip_idx, frame_idx)
                               for clip_idx in range(len(self.videomatte_clips))
                               for frame_idx in range(0, len(self.videomatte_frames[clip_idx]), 10)]
        self.multi = multi
        self.size = size
        self.transform = transform
        self.n_cls = 1+3  # fgr_pha, fgr
        self.set_attrs(total_len=len(self.videomatte_idx))

    def __getitem__(self, idx):
        if random.random() < 0.5:
            bgrs = self._get_random_image_background()
        else:
            bgrs = self._get_random_video_background()

        fgr, fgr_com, sim_pha_m, phas, pha_com, trimap = self._get_videomatte(
            idx)

        fgr, fgr_com, phas, pha_com, bgrs, trimap, sim_pha_m = self.transform(
            fgr, fgr_com, phas, pha_com, bgrs, trimap, sim_pha_m)
        if self.dataset_name == 'BGM':
            im = fgr_com
        elif self.multi:
            if random.random() < 0.8:
                if random.random() < 0.5:
                    im = pha_com*fgr_com+(1-pha_com)*bgrs
                else:
                    im = pha_com*fgr_com
            else:
                im = phas*fgr+(1-phas)*bgrs
        else:
            im = phas*fgr+(1-phas)*bgrs

        r = random.random()
        if r < 0.4:
            sim_pha_m = random_erase(sim_pha_m, max_k_size=13, fill_with=0)
        elif r > 0.6:
            sim_pha_m = random_erase(sim_pha_m, max_k_size=13, fill_with=1)
        else:
            sim_pha_m = random_erase(sim_pha_m, max_k_size=13, fill_with=-1)

        out = dict(ims=im)
        out['fgrs'] = fgr
        out['sim_pha_m'] = sim_pha_m
        out['phas'] = phas
        out['trimap'] = trimap

        return out

    def _get_random_image_background(self):
        with Image.open(os.path.join(self.background_image_dir, random.choice(self.background_image_files))) as bgr:
            bgr = self._downsample_if_needed(bgr.convert('RGB'))
        return bgr

    def _get_random_video_background(self):
        clip_idx = random.choice(range(len(self.background_video_clips)))
        frame_count = len(self.background_video_frames[clip_idx])
        frame_idx = random.choice(range(max(1, frame_count)))
        clip = self.background_video_clips[clip_idx]
        frame = self.background_video_frames[clip_idx][frame_idx]
        with Image.open(os.path.join(self.background_video_dir, clip, frame)) as bgr:
            bgr = self._downsample_if_needed(bgr.convert('RGB'))
        return bgr

    def _get_videomatte(self, idx):
        clip_idx, frame_idx = self.videomatte_idx[idx]
        clip = self.videomatte_clips[clip_idx]
        frame_count = len(self.videomatte_frames[clip_idx])
        frame_idx = (frame_idx + random.randint(0, 10)) % frame_count
        frame_tgt_idx = random.choice(range(max(1, frame_count)))
        frame = self.videomatte_frames[clip_idx][frame_idx]
        frame_tgt = self.videomatte_frames[clip_idx][frame_tgt_idx]
        with Image.open(os.path.join(self.videomatte_dir, 'fgr', clip, frame)) as fgr, \
                Image.open(os.path.join(self.videomatte_dir, 'fgr_com', clip, frame)) as fgr_com, \
                Image.open(os.path.join(self.videomatte_dir, 'pha', clip, frame)) as pha, \
                Image.open(os.path.join(self.videomatte_dir, 'pha_com', clip, frame)) as pha_com:
            if self.split == 'train':
                trimap = Image.open(os.path.join(
                    self.videomatte_dir, 'trimap', clip, frame))
            else:
                trimap = pha
            fgr = self._downsample_if_needed(fgr.convert('RGB'))
            try:
                tgt = self._downsample_if_needed(tgt.convert('RGB'))
            except:
                print('######## Frame have problems: ', os.path.join(
                    self.videomatte_dir, 'fgr', clip, frame_tgt))
                exit(200)
            fgr_com = self._downsample_if_needed(fgr_com.convert('RGB'))
            pha = self._downsample_if_needed(pha.convert('L'))
            pha_com = self._downsample_if_needed(pha_com.convert('L'))
            trimap = self._downsample_if_needed(trimap.convert('L'))
            sim_pha_m = Image.fromarray(
                erode(np.array(pha), random.randint(3, 11)))
        return fgr, fgr_com, sim_pha_m, pha, pha_com, trimap

    def _downsample_if_needed(self, img):
        w, h = img.size
        if min(w, h) > self.size:
            scale = self.size / min(w, h)
            w = int(scale * w)
            h = int(scale * h)
            img = img.resize((w, h))
        return img

    @property
    def unwrapped(self):
        return self
