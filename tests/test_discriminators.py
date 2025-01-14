import optax
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from hifigan import (
    DiscriminatorP,
    DiscriminatorS,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    calculate_disc_loss,
    calculate_gan_loss,
    make_step,
)


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def dummy_input():
    # Create a dummy input tensor for testing
    return jnp.ones((1, 1000))  # Example with 1 channel, 1000 time steps


def test_discriminator_p_initialization(rng_key):
    # Initialize DiscriminatorP
    period = 5
    discriminator_p = DiscriminatorP(period=period, key=rng_key)

    assert isinstance(discriminator_p, DiscriminatorP)
    assert len(discriminator_p.layers) == 5  # Should have 5 layers
    assert discriminator_p.period == period


def test_discriminator_s_initialization(rng_key):
    # Initialize DiscriminatorS
    discriminator_s = DiscriminatorS(key=rng_key)

    assert isinstance(discriminator_s, DiscriminatorS)
    assert len(discriminator_s.layers) == 7  # Should have 7 layers


def test_multi_scale_discriminator_initialization(rng_key):
    # Initialize MultiScaleDiscriminator
    multi_scale_disc = MultiScaleDiscriminator(key=rng_key)

    assert isinstance(multi_scale_disc, MultiScaleDiscriminator)
    assert len(multi_scale_disc.discriminators) == 3  # Should have 3 discriminators


def test_multi_period_discriminator_initialization(rng_key):
    # Initialize MultiPeriodDiscriminator
    multi_period_disc = MultiPeriodDiscriminator(key=rng_key)

    assert isinstance(multi_period_disc, MultiPeriodDiscriminator)
    assert len(multi_period_disc.discriminators) == 5  # Should have 5 discriminators


# def test_discriminator_forward_pass(rng_key, dummy_input):
#     # Initialize a DiscriminatorP and test the forward pass
#     period = 5
#     discriminator_p = DiscriminatorP(period=period, key=rng_key)
#     output, _ = discriminator_p(dummy_input)

#     assert output.shape == (1, 22016)  # Assuming the output has shape (1, 22016)


# def test_discriminator_s_forward_pass(rng_key, dummy_input):
#     # Initialize a DiscriminatorS and test the forward pass
#     discriminator_s = DiscriminatorS(key=rng_key)
#     output, _ = discriminator_s(dummy_input)

# assert output.shape == (1, 22016)  # Assuming the output has shape (1, 22016)


if __name__ == "__main__":
    pytest.main()
