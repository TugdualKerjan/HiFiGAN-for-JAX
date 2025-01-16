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
    wav = np.array(wav)
    wav = np.expand_dims(wav, 0)
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=22050,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        power=2,
        window="hann",
        n_mels=80,
        fmax=8000,
        fmin=0,
    )
    mel = jnp.squeeze(mel, 0)
    mel = jnp.log(jnp.clip(mel, min=1e-5))  # Spectral normalization
    return mel
