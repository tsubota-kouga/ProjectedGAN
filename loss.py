
import math
from typing import Any, List, Optional, Union, Tuple

import torch
from torch import autograd, nn, cuda
from torch.nn import functional as F

from utils import SimCLRAugmentation


def _gradiend_penalty(
        discriminator: nn.Module,
        fake: torch.Tensor,
        real: torch.Tensor,
        eps: Optional[float] = None, scaler=None):
    batch_size = fake.shape[0]
    weight = torch.rand(
            batch_size, 1, 1, 1,
            dtype=fake.dtype,
            device=fake.device)
    interpolate = (fake * (1 - weight) + real * weight).requires_grad_(True)
    disc_interpolate = discriminator(interpolate)
    if scaler is not None:
        disc_interpolate = scaler.scale(disc_interpolate)
        inv_scale = 1. / (scaler.get_scale() + eps)
    grad = autograd.grad(
        outputs=disc_interpolate.sum(),
        inputs=interpolate,
        grad_outputs=torch.ones_like(disc_interpolate),
        create_graph=True)[0]
    if scaler is not None:
        grad = grad * inv_scale
    grad_penalty_loss = (grad.view(batch_size, -1)
                             .norm(2, dim=1) - 1).square().mean()
    return grad_penalty_loss


def _gradiend_penalty_for_multi_scale(
        discriminator,
        fakes, reals,
        eps: Optional[float] = None,
        scaler=None):
    batch_size = fakes[0].shape[0]
    weight = torch.rand(
            batch_size, 1, 1, 1,
            dtype=fakes[0].dtype,
            device=fakes[0].device)
    interpolates = [fake * (1 - weight) + real * weight
                    for fake, real in zip(fakes, reals)]
    interpolates = list(map(lambda x: x.requires_grad_(True), interpolates))
    disc_interpolates = discriminator(interpolates)
    if scaler is not None:
        disc_interpolates = scaler.scale(disc_interpolates)
        inv_scale = 1. / (scaler.get_scale() + eps)
    grad = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True)
    grad_penalty_loss = 0.0
    for g in grad:
        if scaler is not None:
            g = g * inv_scale
        grad_penalty_loss += (g.view(batch_size, -1)
                               .norm(2, dim=1) - 1).square().mean()
    grad_penalty_loss /= len(grad)
    return grad_penalty_loss


def gradiend_penalty(discriminator, fakes, reals, scaler, eps: Optional[float] = None):
    return _gradiend_penalty(discriminator, fakes, reals, eps, scaler)


class R1Regularization(nn.Module):
    def __init__(self,
                 batch_size: int,
                 eps: Optional[float] = None,
                 resolution: Optional[int] = None,
                 use_contrastive_discriminator: bool = False):
        super(R1Regularization, self).__init__()
        self.batch_size = batch_size
        self.eps = eps
        self.use_contrastive_discriminator = use_contrastive_discriminator
        if use_contrastive_discriminator:
            assert resolution is not None
            self.transform = SimCLRAugmentation(resolution=resolution)

    def forward(self,
                reals: torch.Tensor,
                discriminator: Optional[nn.Module] = None,
                reals_out: Optional[torch.Tensor] = None,
                scaler: Optional[cuda.amp.GradScaler] = None) -> torch.Tensor:
        '''
        NOTE: if reals_out is not None (do NOT re-compute), batch_size is same as reals.
        '''
        if reals_out is None or self.use_contrastive_discriminator:  # re-compute
            reals = reals[:self.batch_size]
            reals_out, feature = discriminator(reals, r1_regularize=True)

        if scaler is not None:
            reals_out = scaler.scale(reals_out)
            inv_scale = 1. / (scaler.get_scale() + self.eps)
        grad = autograd.grad(
            outputs=reals_out.sum(),
            inputs=reals.requires_grad_(True)
                if not self.use_contrastive_discriminator else feature,
            create_graph=True)[0]
        if scaler is not None:
            grad = grad * inv_scale
        r1 = grad.flatten(start_dim=1).square().sum(dim=1).mean()
        return r1

def r1_regularization(
    discriminator: nn.Module,
    reals: torch.Tensor,
    eps: float,
    reals_out: Optional[torch.Tensor] = None,
    scaler: Optional[cuda.amp.GradScaler] = None):

    if reals_out is None:  # re-compute
        # reals.requires_grad = True
        reals_out = discriminator(reals, r1_regularize=True)

    if scaler is not None:
        reals_out = scaler.scale(reals_out)
        inv_scale = 1. / (scaler.get_scale() + eps)
    grad = autograd.grad(
        outputs=reals_out.sum(),
        inputs=reals,
        create_graph=True)[0]
    if scaler is not None:
        grad = grad * inv_scale
    r1 = grad.flatten(start_dim=1).square().sum(dim=1).mean()
    return r1


class PathLengthRegularization(nn.Module):
    def __init__(self,
                 batch_size: int,
                 eps: Optional[float] = None,
                 coefficient: float = 0.99,
                 pl_param: Optional[float] = None):
        super(PathLengthRegularization, self).__init__()
        self.register_buffer("moving_average", torch.zeros([1]))
        self.batch_size = batch_size
        self.coefficient = coefficient
        self.eps = eps
        self.pl_param = pl_param

    def forward(self,
                fakes: torch.Tensor,
                latents: List[torch.Tensor],
                scaler: Optional[cuda.amp.GradScaler] = None) -> torch.Tensor:
        fakes = fakes[:self.batch_size]
        noise = torch.randn_like(fakes) / math.sqrt(math.prod(fakes.shape[-2:]))
        outputs = (noise * fakes).sum()

        if self.pl_param is None:
            resolution = fakes.shape[-1]
            pl_param = math.log(2) / ((resolution ** 2) * (math.log(resolution) - math.log(2)))
        else:
            pl_param = self.pl_param

        if scaler is not None:
            outputs = scaler.scale(outputs)
            inv_scale = 1. / (scaler.get_scale() + self.eps)
        grad = autograd.grad(
                outputs=outputs,
                inputs=latents,
                create_graph=True)[0]
        if scaler is not None:
            grad = grad * inv_scale
        path_length = grad.square().sum(dim=1).mean().sqrt()
        penalty = (self.moving_average - path_length).square().mean()
        if not path_length.isnan().any():
            self.moving_average.data = (
                    path_length.mean() * (1 - self.coefficient) +
                    self.moving_average * self.coefficient)
        return penalty * pl_param


class LSGANDiscriminatorLoss(nn.Module):
    def __init__(self,
                 gp_param: Optional[float],
                 drift_param: Optional[float],
                 a: float = 0.0, b: float = 1.0,
                 eps: Optional[float] = None):
        super(LSGANDiscriminatorLoss, self).__init__()
        self.gp_param = gp_param
        self.drift_param = drift_param
        self.a = a
        self.b = b
        self.eps = eps

    def forward(self,
                fakes: Tuple[torch.Tensor, ...],
                reals: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        lsgan_loss = sum([
            0.5 * (_reals_out - self.b).square().mean() + \
            0.5 * (_fakes_out - self.a).square().mean()
            for _reals_out, _fakes_out in zip(reals_out, fakes_out)])
        return lsgan_loss, grad_penalty_loss, drift_loss


class LSGANGeneratorLoss(nn.Module):
    def __init__(self, c: float = 1.0):
        super(LSGANGeneratorLoss, self).__init__()
        self.c = c

    def forward(self,
                fakes_out: Tuple[torch.Tensor, ...],
                reals_out: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        return sum([
            0.5 * (_fakes_out - self.c).square().mean()
            for _fakes_out in fakes_out])


class HingeGeneratorLoss(nn.Module):
    def forward(self,
                fakes_out: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        return sum([-_fakes_out.mean() for _fakes_out in fakes_out])

class HingeDiscriminatorLoss(nn.Module):
    def forward(self,
                fakes_out: Tuple[torch.Tensor, ...],
                reals_out: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        return sum([
            F.relu(1. - _reals_out).mean() + F.relu(1. + _fakes_out).mean()
            for _reals_out, _fakes_out in zip(reals_out, fakes_out)])


class RelativisticAverageHingeDiscriminatorLoss(nn.Module):
    def forward(self,
                fakes_out: Tuple[torch.Tensor, ...],
                reals_out: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        return sum([
            F.relu(1.0 - (_reals_out - _fakes_out.mean())).mean() + \
            F.relu(1.0 + (_fakes_out - _reals_out.mean())).mean()
            for _reals_out, _fakes_out in zip(reals_out, fakes_out)])

class RelativisticAverageHingeGeneratorLoss(nn.Module):
    def forward(self,
                fakes_out: Tuple[torch.Tensor, ...],
                reals_out: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        return sum([
            F.relu(1.0 + (_reals_out - _fakes_out.mean())).mean() + \
            F.relu(1.0 - (_fakes_out - _reals_out.mean())).mean()
            for _reals_out, _fakes_out in zip(reals_out, fakes_out)])


class NonSaturatingGeneratorLoss(nn.Module):
    def forward(self,
                fakes_out: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        loss = sum([F.softplus(-_fakes_out).mean() for _fakes_out in fakes_out])
        return loss


class NonSaturatingDiscriminatorLoss(nn.Module):
    def forward(self,
                fakes_out: Tuple[torch.Tensor, ...],
                reals_out: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        real_loss = sum([F.softplus(-_reals_out).mean() for _reals_out in reals_out])
        fake_loss = sum([F.softplus(_fakes_out).mean() for _fakes_out in fakes_out])

        return real_loss + fake_loss

