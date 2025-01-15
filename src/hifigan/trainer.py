import equinox as eqx
import jax


@eqx.filter_value_and_grad
def calculate_gan_loss(gan, period, scale, x, y):
    gan_result = jax.vmap(gan)(x)[:, :, :22016]  # TODO check this magic number
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

    return G_loss + 45 * l1_loss  # TODO add config for the 30


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
    print(type(x))
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
