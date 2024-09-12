# Copyright (C) 2024 Jiang Xin
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from jittor_matter.data.imagematte import ImageMatteDataset,ImageMatteDatasetSim
from jittor_matter.data.videomatte import VideoMatteDataset,VideoMatteDatasetSim

from jittor_matter.data import augmentation as A
from jittor import transform as T

def create_sim_dataset(dataset_kwargs):
    dataset_kwargs = dataset_kwargs.copy()
    dataset_name = dataset_kwargs.pop("dataset")
    batch_size = dataset_kwargs.pop("batch_size")
    num_workers = dataset_kwargs.pop("num_workers")
    split = dataset_kwargs.pop("split")
    task= dataset_kwargs.pop("type")

    im_size=dataset_kwargs['image_size']

    # load dataset_name
    if dataset_name == "videomatte":
        videomatte_dir=dataset_kwargs['train_data_path'] if split=='train' else dataset_kwargs['test_data_path']
        transform=A.PairCompose([
                                    A.PairRandomAffineAndResize((im_size, im_size), degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.4, 1), shear=(-5, 5)),
                                    A.PairRandomHorizontalFlip(),
                                    A.PairRandomBoxBlur(0.1, 5),
                                    A.PairRandomSharpen(0.1),
                                    A.PairApplyOnlyAtIndices([1], T.ColorJitter(0.15, 0.15, 0.15, 0.05)),
                                    A.PairApply(T.ToTensor())
                                ]) if split=='train' else A.PairCompose([A.PairApply(T.Resize((im_size,im_size))),A.PairApply(T.ToTensor())])
        dataset = VideoMatteDatasetSim(
                videomatte_dir=videomatte_dir,
                background_image_dir=dataset_kwargs['background_images'],
                background_video_dir=dataset_kwargs['background_videos'],
                size=im_size,
                transform=transform,
                multi=True if task=='multi' else False,
                split=split
                )
    elif dataset_name == "BGM":
        videomatte_dir=dataset_kwargs['train_data_path'] if split=='train' else dataset_kwargs['test_data_path']
        transform=A.PairCompose([
                                    A.PairRandomAffineAndResize((im_size, im_size), degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.4, 1), shear=(-5, 5)),
                                    A.PairRandomHorizontalFlip(),
                                    A.PairRandomBoxBlur(0.1, 5),
                                    A.PairRandomSharpen(0.1),
                                    A.PairApplyOnlyAtIndices([1], T.ColorJitter(0.15, 0.15, 0.15, 0.05)),
                                    A.PairApply(T.ToTensor())
                                ]) if split=='train' else A.PairCompose([A.PairApply(T.Resize((im_size,im_size))),A.PairApply(T.ToTensor())])
        dataset = VideoMatteDatasetSim(
                videomatte_dir=videomatte_dir,
                background_image_dir=dataset_kwargs['background_images'],
                background_video_dir=dataset_kwargs['background_videos'],
                size=im_size,
                transform=transform,
                multi=True if task=='multi' else False,
                split=split,
                dataset_name='BGM'
                )
    elif dataset_name == "imagematte":
        image_dir=dataset_kwargs['train_data_path'] if split=='train' else dataset_kwargs['test_data_path']
        transform=A.PairCompose([
                                    A.PairRandomAffineAndResize((im_size, im_size), degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.4, 1), shear=(-5, 5)),
                                    A.PairRandomHorizontalFlip(),
                                    A.PairRandomBoxBlur(0.1, 5),
                                    A.PairRandomSharpen(0.1),
                                    A.PairApplyOnlyAtIndices([1], T.ColorJitter(0.15, 0.15, 0.15, 0.05)),
                                    A.PairApply(T.ToTensor())
                                ]) if split=='train' else A.PairCompose([A.PairApply(T.Resize((im_size,im_size))),A.PairApply(T.ToTensor())])
        dataset = ImageMatteDatasetSim(
                image_dir=image_dir,
                background_image_dir=dataset_kwargs['background_images'],
                background_video_dir=dataset_kwargs['background_videos'],
                size=im_size,
                transform=transform,
                multi=True if task=='multi' else False,
                split=split
                )
    else:
        raise ValueError(f"Dataset {dataset_name} is unknown.")

    loader=dataset.set_attrs(batch_size=batch_size, num_workers = num_workers, shuffle=True, buffer_size=2*(536870912*2))
    return loader



def create_dataset(dataset_kwargs):
    dataset_kwargs = dataset_kwargs.copy()
    dataset_name = dataset_kwargs.pop("dataset")
    batch_size = dataset_kwargs.pop("batch_size")
    num_workers = dataset_kwargs.pop("num_workers")
    split = dataset_kwargs.pop("split")
    task= dataset_kwargs.pop("type")

    im_size=dataset_kwargs['image_size']

    # load dataset_name
    if dataset_name == "videomatte":
        videomatte_dir=dataset_kwargs['train_data_path'] if split=='train' or split=='val' else dataset_kwargs['test_data_path']
        transform=A.PairCompose([
                                    A.PairRandomAffineAndResize((im_size, im_size), degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.4, 1), shear=(-5, 5)),
                                    A.PairRandomHorizontalFlip(),
                                    A.PairRandomBoxBlur(0.1, 5),
                                    A.PairRandomSharpen(0.1),
                                    A.PairApplyOnlyAtIndices([1], T.ColorJitter(0.15, 0.15, 0.15, 0.05)),
                                    A.PairApply(T.ToTensor())
                                ]) if split=='train' else A.PairCompose([A.PairApply(T.Resize((im_size,im_size))),A.PairApply(T.ToTensor())])
        dataset = VideoMatteDataset(
                videomatte_dir=videomatte_dir,
                background_image_dir=dataset_kwargs['background_images'],
                background_video_dir=dataset_kwargs['background_videos'],
                size=im_size,
                transform=transform,
                multi=True if task=='multi' else False,
                split=split
                )
    elif dataset_name == "BGM":
        videomatte_dir=dataset_kwargs['train_data_path'] if split=='train' else dataset_kwargs['test_data_path']
        transform=A.PairCompose([
                                    A.PairRandomAffineAndResize((im_size, im_size), degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.4, 1), shear=(-5, 5)),
                                    A.PairRandomHorizontalFlip(),
                                    A.PairRandomBoxBlur(0.1, 5),
                                    A.PairRandomSharpen(0.1),
                                    A.PairApplyOnlyAtIndices([1], T.ColorJitter(0.15, 0.15, 0.15, 0.05)),
                                    A.PairApply(T.ToTensor())
                                ]) if split=='train' else A.PairCompose([A.PairApply(T.Resize((im_size,im_size))),A.PairApply(T.ToTensor())])
        dataset = VideoMatteDataset(
                videomatte_dir=videomatte_dir,
                background_image_dir=dataset_kwargs['background_images'],
                background_video_dir=dataset_kwargs['background_videos'],
                size=im_size,
                transform=transform,
                multi=True if task=='multi' else False,
                split=split,
                dataset_name='BGM'
                )
    elif dataset_name == "imagematte":
        image_dir=dataset_kwargs['train_data_path'] if split=='train' else dataset_kwargs['test_data_path']
        transform=A.PairCompose([
                                    A.PairRandomAffineAndResize((im_size, im_size), degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.4, 1), shear=(-5, 5)),
                                    A.PairRandomHorizontalFlip(),
                                    A.PairRandomBoxBlur(0.1, 5),
                                    A.PairRandomSharpen(0.1),
                                    A.PairApplyOnlyAtIndices([1], T.ColorJitter(0.15, 0.15, 0.15, 0.05)),
                                    A.PairApply(T.ToTensor())
                                ]) if split=='train' else A.PairCompose([A.PairApply(T.Resize((im_size,im_size))),A.PairApply(T.ToTensor())])
        dataset = ImageMatteDataset(
                image_dir=image_dir,
                background_image_dir=dataset_kwargs['background_images'],
                background_video_dir=dataset_kwargs['background_videos'],
                size=im_size,
                transform=transform,
                multi=True if task=='multi' else False,
                split=split
                )
    else:
        raise ValueError(f"Dataset {dataset_name} is unknown.")

    loader=dataset.set_attrs(batch_size=batch_size, num_workers = num_workers, shuffle=True, buffer_size=2*(536870912*2))
    return loader
