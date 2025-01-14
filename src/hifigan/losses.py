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

    return G_loss + 30 * l1_loss  # TODO add config for the 30


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
