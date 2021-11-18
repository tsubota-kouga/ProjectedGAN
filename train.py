
import numpy as np
import argparse
import copy
from datetime import datetime
import os
from typing import Final, Optional

from timm.data import create_transform, resolve_data_config
import torch
from torch import backends, cuda, nn, optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils
import torch_optimizer
import kornia
from tqdm import tqdm, trange

from hyperparam import HyperParam as hp
from loss import (
    LSGANDiscriminatorLoss,
    LSGANGeneratorLoss,
    NonSaturatingDiscriminatorLoss,
    NonSaturatingGeneratorLoss,
    PathLengthRegularization,
    R1Regularization,
    RelativisticAverageHingeDiscriminatorLoss,
    RelativisticAverageHingeGeneratorLoss,
    HingeGeneratorLoss,
    HingeDiscriminatorLoss,
)
from network import Discriminator, Generator, Projector
from utils import (
    FFHQDataset,
    adjust_dynamic_range,
    load_generator,
    preprocess,
    update_average,
    logit_sigmoid,
)


if __name__ == "__main__":
    backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", help="execute preprocess", action="store_true")
    parser.add_argument("-p", "--path", help="load model", type=str, default=None)
    args = parser.parse_args()

    if args.preprocess:
        os.mkdir("./datasets")
        preprocess(hp.dataset_path[hp.dataset], hp.max_level, hp.dataroot, multi_resolution=False)

    @torch.jit.script
    def random_brightness(x: torch.Tensor) -> torch.Tensor:
        return x + torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5
    @torch.jit.script
    def random_saturation(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2.) + mean
    @torch.jit.script
    def random_contrast(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + mean
    # data_augmentation = nn.Sequential(
    #     transforms.RandomApply([
    #         kornia.augmentation.RandomAffine(
    #             degrees=0,
    #             translate=(0.125, 0.125),
    #             same_on_batch=False,
    #             padding_mode="reflection"),
    #         kornia.augmentation.RandomErasing(same_on_batch=False),
    #         transforms.Lambda(lambd=random_brightness),
    #         transforms.Lambda(lambd=random_saturation),
    #         transforms.Lambda(lambd=random_contrast),
    #         ], p=0.2)
    #     ).to(hp.device)
    # data_augmentation.eval()

    def init_weights(m: nn.Module):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, mean=0., std=0.02)

    projector = Projector(
        feature_channels=hp.feature_channels,
        projected_channels=hp.projected_channels).to(hp.device)
    projector.eval()
    discriminator = Discriminator(
        projected_channels=hp.projected_channels,
        channel_info=hp.discriminator_channels_info
        ).to(hp.device)
    discriminator.train()
    discriminator.apply(init_weights)

    generator = Generator(
        channels_info=hp.generator_channels_info,
        latent_dim=hp.latent_dim
        ).to(hp.device)
    generator.train()
    generator.apply(init_weights)

    dataset = FFHQDataset(
        root=hp.dataroot,
        multi_resolution=False,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]),
        use_fp16=hp.use_fp16)

    dataloader = data.DataLoader(
        dataset,
        batch_size=hp.batch_sizeD * hp.gradient_accumulation,
        shuffle=True,
        pin_memory=True,
        num_workers=16)

    if hp.reload:
        generator, projector, discriminator = \
            load_generator(
                generator,
                projector,
                discriminator,
                path=args.path)

    criterionG = NonSaturatingGeneratorLoss()
    criterionD = NonSaturatingDiscriminatorLoss()

    optimizerG = optim.Adam(
        generator.parameters(),
        lr=hp.glr,
        eps=hp.eps,
        betas=hp.betas,
        weight_decay=hp.weight_decay,
        amsgrad=True)
    optimizerD = optim.Adam(
        discriminator.parameters(),
        lr=hp.dlr,
        eps=hp.eps,
        betas=hp.betas,
        weight_decay=hp.weight_decay,
        amsgrad=True)

    scaler: Final[Optional[cuda.amp.GradScaler]] = \
            cuda.amp.GradScaler() if hp.use_fp16 else None

    writer = SummaryWriter(log_dir=hp.log_dir)


    fixed_noise = torch.randn(
            16, hp.latent_dim, 1 , 1,
            dtype=torch.float16 if hp.use_fp16 else torch.float32,
            device=hp.device)

    adaptive_pseudo_augment_prob = 0.
    lambda_r = 0.
    lambda_rs = []

    global_step = 0

    if hp.move_average_rate is not None:
        generator_ = copy.deepcopy(generator)
        generator_.train()
        update_average(generator_, generator, beta=0)

    for epoch in trange(hp.num_epoch):
        for imgs in tqdm(dataloader):
            B = imgs.shape[0]

            if B != hp.batch_sizeD * hp.gradient_accumulation:
                continue

            # Generator train
            fakes_batches = []
            for idx in range(hp.gradient_accumulation):
                with cuda.amp.autocast(enabled=hp.use_fp16):
                    generator.train()
                    noise = torch.randn(
                        hp.batch_sizeG, hp.latent_dim, 1, 1,
                        dtype=torch.float16 if hp.use_fp16 else torch.float32,
                        device=hp.device)
                    fakes = generator(noise)
                    # fakes_batches.append(data_augmentation(fakes))
                    fakes_batches.append(fakes)
            with cuda.amp.autocast(enabled=hp.use_fp16):
                generator.zero_grad(set_to_none=True)
                discriminator.eval()
                lossG_items = []

                for fakes in fakes_batches:
                    discriminator.train()
                    fakes_out = discriminator(*projector(fakes))
                    lossG = criterionG(fakes_out) / hp.gradient_accumulation
                    lossG_items.append(lossG.item())
                    if scaler is not None:
                        scaler.scale(lossG).backward()
                    else:
                        lossG.backward()

                if scaler is not None:  # hp.use_fp16
                    scaler.step(optimizerG)
                    scaler.update()
                else:
                    optimizerG.step()
                del lossG

                if hp.move_average_rate is not None:
                    update_average(generator_, generator, beta=hp.move_average_rate)

                lossG = sum(lossG_items) / hp.gradient_accumulation

            # fakes_batches = list(map(lambda x: x.detach(), fakes_batches))
            del fakes_batches
            # Discriminator train
            fakes_batches = []
            for idx in range(hp.gradient_accumulation):
                with cuda.amp.autocast(enabled=hp.use_fp16):
                    with torch.inference_mode():
                        generator.eval()
                        noise = torch.randn(
                            hp.batch_sizeD, hp.latent_dim, 1, 1,
                            dtype=torch.float16 if hp.use_fp16 else torch.float32,
                            device=hp.device)
                        fakes = generator(noise)
                        with cuda.amp.autocast(enabled=False):
                            # fakes = data_augmentation(fakes.float())
                            fakes_batches.append(fakes)
                        # fakes_batches.append(fakes.cpu())

            imgs_batches = []
            for idx in range(hp.gradient_accumulation):
                _imgs = imgs[idx * hp.batch_sizeD: (idx + 1) * hp.batch_sizeD]
                imgs_batches.append(_imgs)

            with cuda.amp.autocast(enabled=hp.use_fp16):
                discriminator.zero_grad(set_to_none=True)
                discriminator.train()
                lossD_items = []
                for imgs, fakes in zip(imgs_batches, fakes_batches):
                    with torch.inference_mode():
                        generator_.eval()
                        num_mix_fakes = (torch.rand(hp.batch_sizeD) < adaptive_pseudo_augment_prob).sum().item()
                        if num_mix_fakes > 0:
                            noise = torch.randn(
                                num_mix_fakes, hp.latent_dim, 1, 1,
                                dtype=torch.float16 if hp.use_fp16 else torch.float32,
                                device=hp.device)
                            # with cuda.amp.autocast(enabled=False):
                            #     imgs = data_augmentation(imgs.to(hp.device))
                            imgs = torch.cat([imgs[:hp.batch_sizeD - num_mix_fakes].to(hp.device), generator_(noise)])
                        else:
                            imgs = imgs.to(hp.device)
                        projected_imgs = projector(imgs)
                        projected_fakes = projector(fakes)
                    reals_out = discriminator(*projected_imgs)
                    fakes_out =  discriminator(*projected_fakes)
                    lossD = criterionD(fakes_out, reals_out) / hp.gradient_accumulation
                    with torch.inference_mode():
                        if hp.batch_sizeD != num_mix_fakes:
                            for _reals_out in reals_out:
                                _lambda_r = logit_sigmoid(_reals_out[:hp.batch_sizeD - num_mix_fakes]).sign().mean().item()
                                lambda_rs.append(_lambda_r)

                    lossD_items.append(lossD.item())
                    if scaler is not None:
                        scaler.scale(lossD).backward()
                    else:
                        lossD.backward()

                if scaler is not None:
                    scaler.step(optimizerD)
                    scaler.update()
                else:
                    optimizerD.step()
                del lossD

                lossD = sum(lossD_items) / hp.gradient_accumulation

                if (global_step + 1) % hp.AdaptivePeudoAugment.iteration_per == 0 and \
                        epoch >= hp.AdaptivePeudoAugment.start_epoch:
                    if len(lambda_rs) != 0:
                        lambda_r = sum(lambda_rs) / len(lambda_rs)
                        adaptive_pseudo_augment_prob += \
                            np.sign(lambda_r - hp.AdaptivePeudoAugment.threshold) * \
                            hp.AdaptivePeudoAugment.speed * \
                            hp.batch_sizeD * hp.gradient_accumulation * hp.AdaptivePeudoAugment.iteration_per
                        adaptive_pseudo_augment_prob = np.clip(adaptive_pseudo_augment_prob, 0., hp.AdaptivePeudoAugment.max_prob)
                        lambda_rs = []

            # LOG
            def evaluate():
                writer.add_scalar(
                    "discriminator loss",
                    lossD,
                    global_step)
                writer.add_scalar(
                    "generator loss",
                    lossG,
                    global_step)
                writer.add_scalar(
                    "lambda r",
                    lambda_r,
                    global_step)
                writer.add_scalar(
                    "augmentation p",
                    adaptive_pseudo_augment_prob,
                    global_step)
                with torch.inference_mode():
                    with cuda.amp.autocast(enabled=hp.use_fp16):
                        if hp.move_average_rate is not None:
                            generator_.eval()
                            fakes = generator_(fixed_noise)
                        else:
                            generator.eval()
                            fakes = generator(fixed_noise)
                        fakes = adjust_dynamic_range(fakes)
                    writer.add_image(
                        f"real img",
                        utils.make_grid(
                            imgs_batches[-1].float(),
                            nrow=4,
                            padding=1,
                            normalize=True),
                        global_step)

                    fakes = fakes.float().detach().cpu()
                    writer.add_image(
                        f"generated imgs {hp.resolution}x{hp.resolution}",
                        utils.make_grid(
                            fakes,
                            nrow=4,
                            padding=1,
                            normalize=True),
                        global_step)
            if global_step % 50 == 0:
                evaluate()
                # cuda.empty_cache()

            if global_step % 1000 == 0:
                t = datetime.today()
                torch.save({
                    "generator": generator.state_dict(),
                    "generator_": hp.move_average_rate and generator_.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "projector": projector.state_dict(),
                    "optimizerG": optimizerG.state_dict(),
                    "optimizerD": optimizerD.state_dict(),
                }, os.path.join(hp.model_dir, f"{t.date()}-{t.hour}.model"))
            global_step += 1

    writer.close()
