import equinox as eqx
import equinox.nn as nn
import jax

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
