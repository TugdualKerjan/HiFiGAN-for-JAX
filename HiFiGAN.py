import datetime
import json
import os

import datasets
import equinox as eqx
import jax

os.makedirs("/tmp/jax_cache", exist_ok=True)

# os.environ["XLA_FLAGS"] = (
#     "--xla_gpu_enable_persistent_cache=true --xla_gpu_cache_dir=/tmp/jax_cache"
# )

# Configure JAX caching
jax.config.update("jax_enable_compilation_cache", True)
jax.config.update("jax_persistent_cache_enable_xla_caches", "all")  # Cache everything
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)  # Cache everything
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

import librosa
import optax
from librosa.util import normalize
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


from hifigan import (
    Generator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    make_step,
    mel_spec_base_jit,
)


# from jax import config
# import logging

# Enable both compilation logging and shape checking
# config.update("jax_log_compiles", True)  # Basic logging

# # Turn off most JAX logging
# logging.getLogger("jax").setLevel(logging.ERROR)


# # Custom filter to only show shapes
# class ShapeFilter(logging.Filter):
#     def filter(self, record):
#         return "Compiling" in record.msg and "shapes and types" in record.msg


# # Set up logger for shapes
# shape_logger = logging.getLogger("jax._src.interpreters.pxla")
# shape_logger.addFilter(ShapeFilter())
# shape_logger.setLevel(logging.WARNING)

SAMPLE_RATE = 22050
SEGMENT_SIZE = 8192
RANDOM = jax.random.key(0)
LEARNING_RATE = 2e-4
CHECKPOINT_PATH = "checkpoints"
INIT_FROM = "scratch"

N_EPOCHS = 1
BATCH_SIZE = 16


def transform(sample):
    """Based off the original code that can be found here: https://github.com/jik876/hifi-gan/blob/master/meldataset.py

    Args:
        sample (dict): dict entry in HF Dataset

    Returns:
        dict: updated entry
    """
    k = jax.random.key(0)
    # global RANDOM, SAMPLE_RATE

    # RANDOM, k = jax.random.split(RANDOM)
    wav = sample["audio"]["array"]
    if sample["audio"]["sampling_rate"] != SAMPLE_RATE:
        librosa.resample(wav, sample["audio"]["sampling_rate"], SAMPLE_RATE)
    wav = normalize(wav) * 0.95
    if wav.shape[0] >= SEGMENT_SIZE:
        max_audio_start = wav.shape[0] - SEGMENT_SIZE
        audio_start = jax.random.randint(k, (1,), 0, max_audio_start)[0]
        wav = wav[audio_start : audio_start + SEGMENT_SIZE]

    wav = np.expand_dims(wav, 0)
    mel = mel_spec_base_jit(wav=wav)
    # print(mel.shape)
    return {"mel": mel, "audio": wav, "sample_rate": SAMPLE_RATE}


lj_speech_data = datasets.load_dataset("keithito/lj_speech", trust_remote_code=True)

# lj_speech_data = lj_speech_data.map(transform)
# lj_speech_data.save_to_disk("transformed_lj_speech")
lj_speech_data = datasets.load_from_disk("transformed_lj_speech")
lj_speech_data = lj_speech_data.with_format("jax")
# print(lj_speech_data["train"]["mel"][0].shape)

lj_speech_data = lj_speech_data["train"].train_test_split(0.01)

train_data, eval_data = lj_speech_data["train"], lj_speech_data["test"]

# data = data.filter(lambda x: x["label"] == 1)
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
# learning_rate_schedule = optax.warmup_cosine_decay_schedule(
#     init_value=0.0,
#     peak_value=LEARNING_RATE,
#     warmup_steps=100,
#     decay_steps=N_EPOCHS * (train_data.num_rows // BATCH_SIZE) + 100,
# )

optim1 = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4, b1=0.8, b2=0.99))

gan_optim = optim1.init(generator)  # type: ignore

optim2 = optax.adam(1e-4)
scale_optim = optim2.init(discriminator_s)  # type: ignore

optim3 = optax.adam(1e-4)
period_optim = optim3.init(discriminator_p)  # type: ignore

writer = SummaryWriter(
    log_dir="./runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


with jax.profiler.trace("./tmp/jax-trace"):
    for epoch in range(starting_epoch, N_EPOCHS):
        # permutation = jax.random.permutation(jax.random.key(epoch), jnp.arange(0))
        # for i in tqdm(range(train_data.num_rows // BATCH_SIZE)):
        for i, batch in tqdm(enumerate(train_data.iter(batch_size=BATCH_SIZE))):
            with jax.profiler.StepTraceAnnotation("step", step_num=i):
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
                writer.add_figure(
                    "generated/y_hat_spec",
                    plot_spectrogram(np.array(output[0])),
                    step,
                )
                writer.add_figure(
                    "generated/y_spec",
                    plot_spectrogram(np.array(mels[0])),
                    step,
                )
    eqx.tree_serialise_leaves("./generator.eqx", generator)
