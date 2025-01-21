import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp

LRELU_SLOPE = 0.1


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

    # @jax.jit
    def pad_and_reshape(self, x):
        c, t = x.shape
        n_pad = (self.period - (t % self.period)) % self.period
        x_padded = jnp.pad(x, ((0, 0), (0, n_pad)), mode="reflect")
        t_new = x_padded.shape[-1] // self.period
        return x_padded.reshape(c, t_new, self.period)

    # @jax.jit
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
            nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20, key=key6),
            nn.Conv1d(1024, 1024, 5, 1, padding=2, key=key7),
        ]
        self.conv_post = nn.Conv1d(1024, 1, 3, 1, padding=1, key=key8)

    @jax.jit
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
            for period, y in zip(
                periods, jax.random.split(key, len(periods)), strict=False
            )
        ]

    def __call__(self, x):
        preds = []
        fmaps = []

        for disc in self.discriminators:
            pred, fmap = disc(x)
            preds.append(pred)
            fmaps.append(fmap)

        return preds, fmaps
