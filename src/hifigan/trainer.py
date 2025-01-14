import equinox as eqx
import jax

from . import calculate_disc_loss, calculate_gan_loss


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
