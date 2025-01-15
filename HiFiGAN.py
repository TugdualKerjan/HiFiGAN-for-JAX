import datetime
import json
import os

import datasets
import equinox as eqx
import jax
import jax.numpy as jnp
import librosa
import numpy as np
import optax
from librosa.util import normalize
from tensorboardX import SummaryWriter

from hifigan import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, make_step

SAMPLE_RATE = 22050
SEGMENT_SIZE = 22016
RANDOM = jax.random.key(0)
LEARNING_RATE = 2e-4
CHECKPOINT_PATH = "checkpoints"
INIT_FROM = "scratch"

N_EPOCHS = 32
BATCH_SIZE = 32


def transform(sample):
    """Based off the original code that can be found here: https://github.com/jik876/hifi-gan/blob/master/meldataset.py

    Args:
        sample (dict): dict entry in HF Dataset

    Returns:
        dict: updated entry
    """
    global RANDOM, SAMPLE_RATE

    RANDOM, k = jax.random.split(RANDOM)
    wav = sample["audio"]["array"]
    if sample["audio"]["sampling_rate"] != SAMPLE_RATE:
        librosa.resample(wav, sample["audio"]["sampling_rate"], SAMPLE_RATE)
    wav = normalize(wav) * 0.95
    if wav.shape[0] >= SEGMENT_SIZE:
        max_audio_start = wav.shape[0] - SEGMENT_SIZE
        audio_start = jax.random.randint(k, (1,), 0, max_audio_start)[0]
        wav = wav[audio_start : audio_start + SEGMENT_SIZE]
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
    return {"mel": mel, "audio": wav, "sample_rate": SAMPLE_RATE}


lj_speech_data = datasets.load_dataset("keithito/lj_speech", trust_remote_code=True)

lj_speech_data = lj_speech_data.map(transform)

lj_speech_data = lj_speech_data.with_format("jax")

lj_speech_data = lj_speech_data["train"].train_test_split(0.01)

train_data, eval_data = lj_speech_data["train"], lj_speech_data["test"]


k1, k2, k3 = jax.random.split(RANDOM, 3)
generator = Generator(channels_in=80, channels_out=1, key=k1)
discriminator_s = MultiScaleDiscriminator(key=k2)
discriminator_p = MultiPeriodDiscriminator(key=k3)

os.makedirs(CHECKPOINT_PATH, exist_ok=True)

if INIT_FROM == "scratch":
    print("Starting training from scratch")

    # current_step = 0
    starting_epoch = 0
elif INIT_FROM == "resume":
    print(f"Resuming training from {CHECKPOINT_PATH}")

    def load(filename):
        with open(filename, "rb") as f:
            checkpoint_state = json.loads(f.readline().decode())
            return (
                eqx.tree_deserialise_leaves(f, generator),
                eqx.tree_deserialise_leaves(f, discriminator_p),
                eqx.tree_deserialise_leaves(f, discriminator_s),
                checkpoint_state,
            )

    model, discriminator_p, discriminator_s, checkpoint_state = load(CHECKPOINT_PATH)
    # current_step = checkpoint_state["current_step"]
    current_epoch = checkpoint_state["current_epoch"]

# Define optimizers
optim1 = optax.adam(LEARNING_RATE, b1=0.8, b2=0.99)

gan_optim = optim1.init(generator)  # type: ignore

optim2 = optax.adam(1e-4)
scale_optim = optim2.init(discriminator_s)  # type: ignore

optim3 = optax.adam(1e-4)
period_optim = optim3.init(discriminator_p)  # type: ignore

writer = SummaryWriter(log_dir="./runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

epochs = 100
batch_size = 32

for epoch in range(starting_epoch, N_EPOCHS):
    train_data = train_data.shuffle(seed=epoch)
    for i, batch in enumerate(train_data.iter(batch_size=BATCH_SIZE)):
        # print(batch["audio"][0])
        mels, wavs = batch["mel"], batch["audio"]

        # Terrifying
        (
            gan_loss,
            period_loss,
            scale_loss,
            generator,
            period_disc,
            scale_disc,
            gan_optim,
            period_optim,
            scale_optim,
            output,
        ) = make_step(
            generator,
            discriminator_p,
            discriminator_s,
            mels,
            wavs,
            gan_optim,
            period_optim,
            scale_optim,
            optim1,
            optim2,
            optim3,
        )

        step = epoch * train_data.num_rows + i
        # Log codebook updates to TensorBoard
        writer.add_scalar("Loss/Generator", gan_loss, step)
        writer.add_scalar("Loss/Multi Period", period_loss, step)
        writer.add_scalar("Loss/Multi Scale", scale_loss, step)
