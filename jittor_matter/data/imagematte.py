import os
import random
from jittor.dataset.dataset import Dataset
from PIL import Image

from jittor_matter.data.utils import erode, random_erase
import numpy as np

class ImageMatteDatasetSim(Dataset):
    def __init__(self,
                 image_dir,
                 background_image_dir,
                 background_video_dir,
                 size,
                 transform=None,
                 multi=True,
                 split='train'
                 ):
        super().__init__()
        self.background_image_dir = background_image_dir
        self.background_image_files = [x for x in os.listdir(background_image_dir) if 'checkpoint' not in x][:5000]
        self.background_video_dir = background_video_dir
        self.background_video_clips = sorted(os.listdir(background_video_dir))
        self.background_video_frames = [sorted([x for x in os.listdir(os.path.join(background_video_dir, clip)) if x.endswith('.jpg')])
                                        for clip in self.background_video_clips]
        
        self.image_dir=image_dir
        
        self.imgs=[x for x in sorted(os.listdir(os.path.join(image_dir,'pha'))) if 'checkpoint' not in x]

        self.multi=multi
        self.split=split
        self.size = size
        self.transform = transform

        self.set_attrs(total_len=len(self.imgs))

    
    def __getitem__(self, idx):
        if random.random() < 0.7:
            bgrs = self._get_random_image_background()
        else:
            bgrs = self._get_random_video_background()
        
        fgr, fgr_com, sim_pha_m, phas,pha_com,trimap = self.get_ims(idx)
        
        fgr,fgr_com, phas, pha_com,trimap, bgrs,sim_pha_m = self.transform(fgr,fgr_com, phas, pha_com,trimap, bgrs,sim_pha_m)
        if self.multi:
            im=pha_com*fgr_com+(1-pha_com)*bgrs
        else:
            im=phas*fgr+(1-phas)*bgrs

        r=random.random()
        if r < 0.4:
            sim_pha_m=random_erase(sim_pha_m,max_k_size=13,fill_with=0)
        elif r>0.6:
            sim_pha_m=random_erase(sim_pha_m,max_k_size=13,fill_with=1)
        else:
            sim_pha_m=random_erase(sim_pha_m,max_k_size=13,fill_with=-1)
        
        out = dict(ims=im)
        out['fgrs']=fgr
        out['sim_pha_m']=sim_pha_m
        out['phas']=phas
        out['trimap']=trimap
        
        return out
    
    def get_ims(self,idx):
        im_name=self.imgs[idx]
        with Image.open(os.path.join(self.image_dir, 'fgr_com', im_name)) as fgr_com, \
            Image.open(os.path.join(self.image_dir, 'fgr', im_name)) as fgr, \
            Image.open(os.path.join(self.image_dir, 'pha', im_name)) as pha, \
            Image.open(os.path.join(self.image_dir, 'pha_com', im_name)) as pha_com: 
                if self.split == 'train':
                    trimap = Image.open(os.path.join(self.image_dir, 'trimap', im_name))
                else:
                    trimap=pha
                fgr = self._downsample_if_needed(fgr.convert('RGB'))
                fgr_com = self._downsample_if_needed(fgr_com.convert('RGB'))
                pha = self._downsample_if_needed(pha.convert('L'))
                pha_com = self._downsample_if_needed(pha_com.convert('L'))
                trimap = self._downsample_if_needed(trimap.convert('L'))
                sim_pha_m=Image.fromarray(erode(np.array(pha),random.randint(3,11)))
        return fgr, fgr_com, sim_pha_m, pha ,pha_com,trimap
    
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


class ImageMatteDataset(Dataset):
    def __init__(self,
                 image_dir,
                 background_image_dir,
                 background_video_dir,
                 size,
                 transform=None,
                 multi=True,
                 split='train'
                 ):
        super().__init__()
        self.background_image_dir = background_image_dir
        self.background_image_files = [x for x in os.listdir(background_image_dir) if 'checkpoint' not in x][:5000]
        self.background_video_dir = background_video_dir
        self.background_video_clips = sorted(os.listdir(background_video_dir))
        self.background_video_frames = [sorted([x for x in os.listdir(os.path.join(background_video_dir, clip)) if x.endswith('.jpg')])
                                        for clip in self.background_video_clips]
        
        self.image_dir=image_dir
        
        self.imgs=[x for x in sorted(os.listdir(os.path.join(image_dir,'pha'))) if 'checkpoint' not in x]

        self.multi=multi
        self.split=split
        self.size = size
        self.transform = transform

        self.set_attrs(total_len=len(self.imgs))

    
    def __getitem__(self, idx):
        if random.random() < 0.7:
            bgrs = self._get_random_image_background()
        else:
            bgrs = self._get_random_video_background()
        
        fgr, fgr_com, tgt, phas,pha_com,trimap = self.get_ims(idx)
        
        fgr,fgr_com, phas, pha_com,trimap, bgrs = self.transform(fgr,fgr_com, phas, pha_com,trimap, bgrs)
        tgt = self.transform(tgt)[0]
        if self.multi:
            im=pha_com*fgr_com+(1-pha_com)*bgrs
        else:
            im=phas*fgr+(1-phas)*bgrs
        out = dict(ims=im)
        out['fgrs']=fgr
        out['tgts']=tgt
        out['phas']=phas
        out['trimap']=trimap
        
        return out
    
    def get_ims(self,idx):
        im_name=self.imgs[idx]
        with Image.open(os.path.join(self.image_dir, 'tgt', im_name)) as tgt, \
            Image.open(os.path.join(self.image_dir, 'fgr_com', im_name)) as fgr_com, \
            Image.open(os.path.join(self.image_dir, 'fgr', im_name)) as fgr, \
            Image.open(os.path.join(self.image_dir, 'pha', im_name)) as pha, \
            Image.open(os.path.join(self.image_dir, 'pha_com', im_name)) as pha_com: 
                if self.split == 'train':
                    trimap = Image.open(os.path.join(self.image_dir, 'trimap', im_name))
                else:
                    trimap=pha
                fgr = self._downsample_if_needed(fgr.convert('RGB'))
                tgt = self._downsample_if_needed(tgt.convert('RGB'))
                fgr_com = self._downsample_if_needed(fgr_com.convert('RGB'))
                pha = self._downsample_if_needed(pha.convert('L'))
                pha_com = self._downsample_if_needed(pha_com.convert('L'))
                trimap = self._downsample_if_needed(trimap.convert('L'))
        return fgr, fgr_com, tgt, pha ,pha_com,trimap
    
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


