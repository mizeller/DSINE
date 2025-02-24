"""
Run with: 
CUDA_VISIBLE_DEVICES=0 python test_arctic.py ./experiments/exp001_cvpr2024/dsine.txt

NOTE:
Defaults to processing `arctic_s03_mixer_crop`.
"""

import os
import sys
import cv2
import imageio
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import sys

sys.path.append("../../")
import utils.utils as utils
import projects.dsine.config as config
from utils.projection import intrins_from_fov, intrins_from_txt


def get_writer_cfg(fps: int = 30, is_mask: bool = False, height: int = None):
    """Get standard video writer configuration.
    - if is_mask=True -> lossless, binary video
    - the provided height should be an even number (req. H.264 codec)!
    """
    config = {
        "fps": fps,
        "format": "FFMPEG",  # for .mp4 outputs
        "codec": "libx264",
        "macro_block_size": 1,
        "pixelformat": "yuv420p",
        "mode": "I",
        "output_params": [],
    }
    if is_mask:
        # it's important that the video only contains valid pixel values
        # (i.e. for segmentation masks).
        # only use lossless mode for H.264 if required though (big files)
        config["output_params"].extend(["-crf", "0"])
        config["pixelformat"] = "gray8"
    if height:
        # NOTE: do it this way to ensure that width remains an even number
        config["output_params"].extend(["-vf", f"scale=trunc(oh*a/2)*2:{height}"])

    return config


if __name__ == "__main__":
    device = torch.device("cuda")
    args = config.get_args(test=True)
    assert os.path.exists(args.ckpt_path)

    if args.NNET_architecture == "v00":
        from models.dsine.v00 import DSINE_v00 as DSINE
    elif args.NNET_architecture == "v01":
        from models.dsine.v01 import DSINE_v01 as DSINE
    elif args.NNET_architecture == "v02":
        from models.dsine.v02 import DSINE_v02 as DSINE
    elif args.NNET_architecture == "v02_kappa":
        from models.dsine.v02_kappa import DSINE_v02_kappa as DSINE
    else:
        raise Exception("invalid arch")

    model = DSINE(args).to(device)
    model = utils.load_checkpoint(args.ckpt_path, model)
    model.eval()

    # Define input and output paths

    paths = {
        "video": Path(
            "/home/michel/projects/hold-gs-data/data/arctic_s03_mixer_crop/video.mp4"
        ),
        "data": Path(
            "/home/michel/projects/hold-gs-data/data/arctic_s03_mixer_crop/data.pt"
        ),
    }

    for v in paths.values():
        assert v.exists(), v

    out_p = paths["video"].parent / "normals.mp4"

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Open video file
    cap = cv2.VideoCapture(paths["video"])
    frame_count = 0

    # Get video properties for output video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video with {total_frames} frames...")
    with torch.no_grad():
        with imageio.get_writer(out_p, **get_writer_cfg()) as writer:
            pbar = tqdm(total=total_frames, desc="Processing frames")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert to float and normalize to [0, 1]
                img = frame.astype(np.float32) / 255.0
                img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

                # pad input
                _, _, orig_H, orig_W = img.shape
                lrtb = utils.get_padding(orig_H, orig_W)
                img = F.pad(img, lrtb, mode="constant", value=0.0)
                img = normalize(img)


                # Use default intrinsics with 60-degree FOV
                # TODO: use estimated intrinsics
                intrins = intrins_from_fov(
                    new_fov=60.0, H=orig_H, W=orig_W, device=device
                ).unsqueeze(0)
                intrins[:, 0, 2] += lrtb[0]
                intrins[:, 1, 2] += lrtb[2]

                pred_norm = model(img, intrins=intrins)[-1]
                pred_norm = pred_norm[
                    :, :, lrtb[2] : lrtb[2] + orig_H, lrtb[0] : lrtb[0] + orig_W
                ]

                pred_norm = pred_norm.squeeze().detach().cpu().permute(1, 2, 0).numpy()
                # normal mapping
                pred_norm = (((pred_norm + 1) * 0.5) * 255).astype(np.uint8)

                # update
                writer.append_data(pred_norm)
                pbar.update(1)

            pbar.close()

    cap.release()
    print(f"Processing complete. Output saved to {out_p}")
