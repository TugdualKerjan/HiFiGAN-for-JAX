import equinox as eqx
import jax
import jax.numpy as jnp

from hifigan.utils import mel_spec_base_jit


@eqx.filter_jit(donate="all-except-first")
@eqx.filter_value_and_grad
def calculate_gan_loss(gan, period, scale, x, y):
    gan_result = jax.vmap(gan)(x)  # TODO check this magic number
    # print(gan_result.shape)
    mel_gan_result = jax.vmap(mel_spec_base_jit)(gan_result)

    fake_scale, fake_feature_map_period = jax.vmap(scale)(gan_result)
    fake_period, fake_feature_map_scale = jax.vmap(period)(gan_result)

    _, real_feature_map_period = jax.vmap(scale)(y)
    _, real_feature_map_scale = jax.vmap(period)(y)

    loss = jnp.mean(jax.numpy.abs(mel_gan_result - x)) * 45
    for fake in fake_period:
        loss += jnp.mean((fake - 1) ** 2)
    for fake in fake_scale:
        loss += jnp.mean((fake - 1) ** 2)

    for a, b in zip(fake_feature_map_period, real_feature_map_period, strict=False):
        for c, d in zip(a, b, strict=False):
            loss += jnp.mean(jnp.abs(c - d))

    for a, b in zip(fake_feature_map_scale, real_feature_map_scale, strict=False):
        for c, d in zip(a, b, strict=False):
            loss += jnp.mean(jnp.abs(c - d))

    # G_loss_scale = jax.numpy.mean((fake_scale - jax.numpy.ones(batch_size)) ** 2)

    return loss  # TODO add config for the 30


@eqx.filter_jit
@eqx.filter_value_and_grad
def calculate_disc_loss(model, fake, real):
    fake_result, _ = jax.vmap(model)(fake)
    real_result, _ = jax.vmap(model)(real)
    loss = 0
    for fake_res, real_res in zip(fake_result, real_result, strict=False):
        fake_loss = jnp.mean((fake_res) ** 2)
        real_loss = jnp.mean((real_res - 1) ** 2)
        loss += fake_loss + real_loss

    return loss


@eqx.filter_jit(donate="all")
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
    result = jax.vmap(gan)(x)

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
        jax.vmap(mel_spec_base_jit)(result),
    )
