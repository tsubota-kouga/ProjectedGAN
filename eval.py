
import numpy as np
import sys
import argparse

import cv2
import torch
from torch import cuda, backends
from torchvision import utils

from hyperparam import HyperParam as hp
from network import Generator
from utils import adjust_dynamic_range, load_generator

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None)
    args = parser.parse_args()
    backends.cudnn.benchmark = True
    generator = Generator(
        channels_info=hp.generator_channels_info,
        latent_dim=hp.latent_dim
        ).to(hp.device)
    load_generator(generator, eval_mode=True, path=args.path)
    generator.eval()

    with torch.inference_mode():
        while True:
            noise = torch.randn(
                    18, hp.latent_dim, 1, 1,
                    device=hp.device,
                    dtype=torch.float16 if hp.use_fp16 else torch.float)
            with cuda.amp.autocast(enabled=hp.use_fp16):
                imgs = generator(noise).float().cpu()
                imgs = adjust_dynamic_range(imgs)
            imgs = utils.make_grid(imgs, nrow=6, padding=1, normalize=True)
            imgs = imgs.numpy().transpose([1, 2, 0])
            imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
            cv2.imshow("generated images", imgs)
            cmd = cv2.waitKey(0)
            if cmd == ord("q"):
                cv2.destroyAllWindows()
                break
            elif cmd == ord("c"):
                pass
            else:
                print(f"cmd must be 'q' or 'c', invalid: {cmd}")
                print("default: continue")
            cv2.destroyAllWindows()

