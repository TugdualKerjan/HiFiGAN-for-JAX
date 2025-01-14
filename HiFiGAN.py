#!/usr/bin/env python


import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp

LRELU_SLOPE = 0.1


class ResBlock(eqx.Module):
    conv_dil: list
    conv_straight: list
    norm = nn.WeightNorm

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        key=None,
    ):
        if key is None:
            raise ValueError("The 'key' parameter cannot be None.")
        self.conv_dil = [
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding="SAME",
                key=y,
            )
            for y in jax.random.split(key, 3)
        ]
        self.conv_straight = [
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=1,
                padding="SAME",
                key=y,
            )
            for y in jax.random.split(key, 3)
        ]

    def __call__(self, x):
        for c1, c2 in zip(self.conv_dil, self.conv_straight, strict=False):
            y = jax.nn.leaky_relu(x, LRELU_SLOPE)
            y = self.norm(c1)(y)
            y = jax.nn.leaky_relu(y, LRELU_SLOPE)
            y = self.norm(c2)(y)
            x = y + x

        return x


class MRF(eqx.Module):
    resblocks: list

    def __init__(self, channel_in: int, kernel_sizes: list, dilations: list, key=None):
        if key is None:
            raise ValueError("The 'key' parameter cannot be None.")
        self.resblocks = [
            ResBlock(channel_in, kernel_size, dilation, key=y)
            for kernel_size, dilation, y in zip(
                kernel_sizes,
                dilations,
                jax.random.split(key, len(kernel_sizes)),
                strict=False,
            )
        ]

    def __call__(self, x):
        y = self.resblocks[0](x)
        for block in self.resblocks[1:]:
            y += block(x)

        return y / len(self.resblocks)


class Generator(eqx.Module):
    conv_pre: nn.Conv1d

    layers: list

    post_magic: nn.Conv1d

    norm = nn.WeightNorm

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        h_u=512,
        k_u=(16, 16, 4, 4),
        upsample_rate_decoder=(8, 8, 2, 2),
        k_r=(3, 7, 11),
        dilations=(1, 3, 5),
        key=None,
    ):
        if key is None:
            raise ValueError("The 'key' parameter cannot be None.")
        key, grab = jax.random.split(key, 2)
        self.conv_pre = nn.Conv1d(channels_in, h_u, kernel_size=7, dilation=1, padding=3, key=grab)

        # This is where the magic happens. Upsample aggressively then more slowly.
        # TODO could play around with this.
        # Then convolve one last time (Curious to see the weights to see if has good impact)
        self.layers = [
            (
                nn.ConvTranspose1d(
                    int(h_u / (2**i)),
                    int(h_u / (2 ** (i + 1))),
                    kernel_size=k,
                    stride=u,
                    padding="SAME",
                    key=y,
                ),
                MRF(
                    channel_in=int(h_u / (2 ** (i + 1))),
                    kernel_sizes=k_r,
                    dilations=dilations,
                    key=y,
                ),
            )
            for i, (k, u, y) in enumerate(
                zip(
                    k_u,
                    upsample_rate_decoder,
                    jax.random.split(key, len(k_u)),
                    strict=False,
                )
            )
        ]

        self.post_magic = nn.Conv1d(
            int(h_u / (2 ** len(k_u))),
            channels_out,
            kernel_size=7,
            stride=1,
            padding=3,
            use_bias=False,
            key=key,
        )
        # self.post_magic = nn.WeightNorm(self.post_magic,

    def __call__(self, x):
        y = self.norm(self.conv_pre)(x)

        for upsample, mrf in self.layers:
            y = jax.nn.leaky_relu(y, LRELU_SLOPE)
            y = self.norm(upsample)(y)  # Upsample
            y = mrf(y)

        y = jax.nn.leaky_relu(y, LRELU_SLOPE)
        y = self.norm(self.post_magic)(y)
        y = jax.nn.tanh(y)
        return y


class DiscriminatorP(eqx.Module):
    layers: list
    period: int
    conv_post: nn.Conv2d
    norm = nn.WeightNorm

    def __init__(
        self,
        period: int,
        kernel_size=5,
        stride=3,
        key: jax.Array = None,
    ):
        self.period = period

        keys = jax.random.split(key, 6)
        self.layers = [
            nn.Conv2d(
                1,
                32,
                (kernel_size, 1),
                (stride, 1),
                padding="SAME",
                key=keys[0],
            ),
            nn.Conv2d(
                32,
                128,
                (kernel_size, 1),
                (stride, 1),
                padding="SAME",
                key=keys[1],
            ),
            nn.Conv2d(
                128,
                512,
                (kernel_size, 1),
                (stride, 1),
                padding="SAME",
                key=keys[2],
            ),
            nn.Conv2d(
                512,
                1024,
                (kernel_size, 1),
                (stride, 1),
                padding="SAME",
                key=keys[3],
            ),
            nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding="SAME", key=keys[4]),
        ]
        self.conv_post = nn.Conv2d(1024, 1, (3, 1), 1, padding="SAME", key=keys[5])

    def pad_and_reshape(self, x):
        c, t = x.shape
        n_pad = (self.period - (t % self.period)) % self.period
        x_padded = jnp.pad(x, ((0, 0), (0, n_pad)), mode="reflect")
        t_new = x_padded.shape[-1] // self.period
        return x_padded.reshape(c, t_new, self.period)

    def __call__(self, x):
        # Feature map for loss
        fmap = []

        x = self.pad_and_reshape(x)
        for layer in self.layers:
            x = self.norm(layer)(x)
            x = jax.nn.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.norm(self.conv_post)(x)
        fmap.append(x)
        x = jnp.reshape(x, shape=(1, -1))
        return x, fmap


class DiscriminatorS(eqx.Module):
    layers: list
    conv_post: nn.Conv1d
    norm = nn.WeightNorm

    def __init__(self, key: jax.Array = None):
        key1, key2, key3, key4, key5, key6, key7, key8 = jax.random.split(key, 8)

        self.layers = [
            nn.Conv1d(1, 128, 15, 1, padding=7, key=key1),
            nn.Conv1d(128, 128, 41, 2, groups=4, padding=20, key=key2),
            nn.Conv1d(128, 256, 41, 2, groups=16, padding=20, key=key3),
            nn.Conv1d(256, 512, 41, 4, groups=16, padding=20, key=key4),
            nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20, key=key5),
            nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20, key=key),
            nn.Conv1d(1024, 1024, 5, 1, padding=2, key=key7),
        ]
        self.conv_post = nn.Conv1d(1024, 1, 3, 1, padding=1, key=key8)

    def __call__(self, x):
        # Feature map for loss
        fmap = []

        for layer in self.layers:
            x = self.norm(layer)(x)
            x = jax.nn.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)

        x = self.norm(self.conv_post)(x)
        fmap.append(x)
        x = jax.numpy.reshape(x, shape=(1, -1))

        return x, fmap


class MultiScaleDiscriminator(eqx.Module):
    discriminators: list
    meanpool: nn.AvgPool1d = nn.AvgPool1d(4, 2, padding=2)

    # TODO need to add spectral norm things
    def __init__(self, key: jax.Array = None):
        key1, key2, key3 = jax.random.split(key, 3)

        self.discriminators = [
            DiscriminatorS(key1),
            DiscriminatorS(key2),
            DiscriminatorS(key3),
        ]
        # self.meanpool = nn.AvgPool1d(4, 2, padding=2)

    def __call__(self, x):
        preds = []
        fmaps = []

        for disc in self.discriminators:
            pred, fmap = disc(x)
            preds.append(pred)
            fmaps.append(fmap)
            x = self.meanpool(x)  # Subtle way of scaling things down by 2

        return preds, fmaps


class MultiPeriodDiscriminator(eqx.Module):
    discriminators: list

    def __init__(self, periods=(2, 3, 5, 7, 11), key: jax.Array = None):
        self.discriminators = [
            DiscriminatorP(period, key=y)
            for period, y in zip(periods, jax.random.split(key, len(periods)), strict=False)
        ]

    def __call__(self, x):
        preds = []
        fmaps = []

        for disc in self.discriminators:
            pred, fmap = disc(x)
            preds.append(pred)
            fmaps.append(fmap)

        return preds, fmaps


@eqx.filter_value_and_grad
def calculate_gan_loss(gan, period, scale, x, y):
    gan_result = jax.vmap(gan)(x)[:, :, :22016]
    print(gan_result.shape)
    fake_scale, _ = jax.vmap(scale)(gan_result)
    fake_period, _ = jax.vmap(period)(gan_result)

    l1_loss = jax.numpy.mean(jax.numpy.abs(gan_result - y))  # L1 loss
    G_loss = 0
    for fake in fake_period:
        G_loss += jax.numpy.mean((fake - 1) ** 2)
    for fake in fake_scale:
        G_loss += jax.numpy.mean((fake - 1) ** 2)
    # G_loss_scale = jax.numpy.mean((fake_scale - jax.numpy.ones(batch_size)) ** 2)

    return G_loss + 30 * l1_loss


@eqx.filter_value_and_grad
def calculate_disc_loss(model, fake, real):
    fake_result, _ = jax.vmap(model)(fake)
    real_result, _ = jax.vmap(model)(real)
    loss = 0
    for fake_res, real_res in zip(fake_result, real_result, strict=False):
        fake_loss = jax.numpy.mean((fake_res) ** 2)
        real_loss = jax.numpy.mean((real_res - 1) ** 2)
        loss += fake_loss + real_loss

    return loss


@eqx.filter_jit
def make_step(
    gan,
    period_disc,
    scale_disc,
    x,
    y,
    gan_optim,
    period_optim,
    scale_optim,
    optim1,
    optim2,
    optim3,
):
    result = jax.vmap(gan)(x)[:, :22016]

    # trainable_scale, _ = eqx.partition(scale_disc, eqx.is_inexact_array)
    # trainable_period, _ = eqx.partition(period_disc, eqx.is_inexact_array)

    loss_scale, grads_scale = calculate_disc_loss(scale_disc, result, y)
    updates, scale_optim = optim2.update(grads_scale, scale_optim, scale_disc)
    scale_disc = eqx.apply_updates(scale_disc, updates)

    loss_period, grads_period = calculate_disc_loss(period_disc, result, y)
    updates, period_optim = optim3.update(grads_period, period_optim, period_disc)
    period_disc = eqx.apply_updates(period_disc, updates)

    loss_gan, grads_gan = calculate_gan_loss(gan, period_disc, scale_disc, x, y)
    updates, gan_optim = optim1.update(grads_gan, gan_optim, gan)
    gan = eqx.apply_updates(gan, updates)

    return (
        loss_gan,
        loss_period,
        loss_scale,
        gan,
        period_disc,
        scale_disc,
        gan_optim,
        period_optim,
        scale_optim,
        result,
    )
