import os
import cv2
import numpy as np
from PIL import Image
import jittor as jt
import jittor.transform as transforms
from jittor_matter import config
import click
from pathlib import Path
from jittor_matter.models.factory import create_filter, create_model
# from jittor_matter.eval_metrics_any import eval_metrics
jt.flags.use_cuda = jt.has_cuda
print('******* jt.has_cuda', jt.has_cuda)

'''

'''


def checkdir(p):
    if not os.path.exists(p):
        os.makedirs(p)


jt_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Resize((768,768))
    ]
)


def matting(model, video, result, size=(768, 768), tgt_file=None):
    frames = sorted(os.listdir(video))
    tgt = cv2.imread(tgt_file)
    tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB)
    tgt = cv2.resize(tgt, size, cv2.INTER_AREA)
    tgt_PIL = Image.fromarray(tgt.astype(np.uint8))
    tgt_tensor = jt_transforms(tgt_PIL)
    tgt_tensor = tgt_tensor[None, :, :, :]
    tgt_tensor = jt.Var(tgt_tensor)

    # video writer

    print('Start matting...')
    for frame in frames:
        frame_PIL = Image.open(os.path.join(video, frame))
        frame_PIL = frame_PIL.resize(size)
        frame_tensor = jt_transforms(frame_PIL)
        frame_tensor = frame_tensor[None, :, :, :]
        frame_tensor = jt.Var(frame_tensor)

        with jt.no_grad():
            out = model(frame_tensor, tgt_tensor, inference=True, tm=False)
        matte_tensor = out[0]
        matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
        matte_np = matte_tensor[0].data.transpose(1, 2, 0)*255
        matte_np = matte_np.astype(np.uint8)
        # matte_np = cv2.normalize(matte_np, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(
            result, frame.replace('.jpg', '.png')), matte_np)
        # if alpha_matte:
        #

    print('Save the result video to {0}'.format(result))

# set up configuration


@click.command(help="")
@click.option("--log-dir", type=str, help="logging directory")
@click.option("--dataset", type=str)
@click.option('--video_root', type=str, help='input video file')
@click.option("--im-size", default=None, type=int, help="dataset resize size")
@click.option("--crop-size", default=None, type=int)
@click.option("--token-num", default=16, type=int)
@click.option("--backbone", default="", type=str)
@click.option("--decoder", default="", type=str)
@click.option("--normalization", default=None, type=str)
@click.option("--dropout", default=0.0, type=float)
@click.option("--drop-path", default=0.0, type=float)
@click.option("--task", default="multi", type=str)
@click.option("--tgt-prefix", default="", type=str)
def main(log_dir,
         dataset,
         video_root,
         im_size,
         crop_size,
         token_num,
         backbone,
         decoder,
         normalization,
         dropout,
         drop_path,
         task,
         tgt_prefix
         ):

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = log_dir / "checkpoint.pkl"

    cfg = config.load_config()
    model_cfg = cfg["model"][backbone]
    dataset_cfg = cfg["dataset"][dataset]
    if "mask_transformer" in decoder:
        decoder_cfg = cfg["decoder"]["mask_transformer"]
    else:
        decoder_cfg = cfg["decoder"][decoder]
    if not im_size:
        im_size = dataset_cfg["im_size"]
    if not crop_size:
        crop_size = dataset_cfg.get("crop_size", im_size)

    model_cfg["image_size"] = (crop_size, crop_size)
    model_cfg["backbone"] = backbone
    model_cfg["dropout"] = dropout
    model_cfg["drop_path_rate"] = drop_path
    decoder_cfg["name"] = decoder
    model_cfg["decoder"] = decoder_cfg
    model_cfg["n_cls"] = token_num
    if normalization:
        model_cfg["normalization"] = normalization
    tgt_filter = create_filter(model_cfg)
    model = create_model(tgt_filter, 'ckpt_jittor/tgt_filter.pkl',
                         'ckpt_jittor/matting_model.pkl', load_matter=False)

    print(
        f"###################Resuming training from checkpoint: {checkpoint_path}")
    checkpoint = jt.load(str(checkpoint_path))
    print(f'@@@@@@@ epoch_', checkpoint['epoch'])
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if not os.path.exists(video_root):
        print('Cannot find the input video: {0}'.format(video_root))
        exit()
    video_names = sorted(os.listdir(os.path.join(video_root, 'src')))
    print(video_names)
    save_root = os.path.join(f'Eval_results_{video_root.split("/")[-1]}/')
    checkdir(save_root)

    for v_name in video_names:
        filename_video = os.path.join(video_root, 'src', v_name)

        print('##### video_root:', video_root)
        tgt_file = os.path.join(video_root, 'tgt', tgt_prefix+v_name+'.jpg')
        print("#### tgt_file: ", tgt_file)

        result = os.path.join(save_root, v_name)
        checkdir(result)
        model.init_buffer()
        with jt.no_grad():
            matting(model, filename_video, result, size=(
                im_size, im_size), tgt_file=tgt_file)
    # eval_metrics(pred_dir=save_root,
    #              true_dir='/'.join(video_root.split('/')[:-1]))


if __name__ == '__main__':
    main()

'''
CUDA_VISIBLE_DEVICES=3 python3 -m jittor_matter.evaluate_video_JPEG \
    --video_root ../Dataset/VideoMatte240K_JPEG/test \
    --dataset videomatte   \
    --backbone vit_tiny_patch16_384 \
    --decoder mask_transformer \
    --im-size 512 \
    --task real \
    --log-dir ckpt_jittor \
    --tgt-prefix mat_
'''
