import jax
import jax.numpy as jnp
import librosa
import numpy as np


def mel_spec_base(wav: jax.Array) -> jax.Array:
    """Mel transform, takes in 22050 Hz signal

    Args:
        wav (jax.Array): 1D tensor

    Returns:
        jax.Array: (80, N) tensor
    """
    wav = np.pad(
        wav,
        ((0, 0), (int((1024 - 256) / 2), int((1024 - 256) / 2))),
        mode="reflect",
    )
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=22050,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        power=2,
        window="hann",
        n_mels=80,
        pad_mode="reflect",
        fmax=8000,
        center=False,
        fmin=0,
    )
    mel = jnp.squeeze(mel, 0)
    mel = jnp.log(jnp.clip(mel, min=1e-5))  # Spectral normalization
    return mel
