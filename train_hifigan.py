from collections import defaultdict
import datetime
import json
import os
import time

import datasets
import equinox as eqx
import jax
import jax.numpy as jnp

os.makedirs("/tmp/jax_cache", exist_ok=True)

# os.environ["XLA_FLAGS"] = (
#     "--xla_gpu_enable_persistent_cache=true --xla_gpu_cache_dir=/tmp/jax_cache"
# )

from jax import config

config.update("jax_debug_nans", True)

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

N_EPOCHS = 1000
BATCH_SIZE = 256


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
    return {"mel": np.array(mel), "audio": np.array(wav), "sample_rate": SAMPLE_RATE}


# lj_speech_data = datasets.load_dataset("keithito/lj_speech", trust_remote_code=True)

# lj_speech_data = lj_speech_data.map(transform)

# from datasets import Features, Array2D, Value

# # Define the exact shapes we expect from the transform function
# features = Features(
#     {
#         "mel": Array2D(
#             shape=(80, 32), dtype="float32"
#         ),  # From mel_spec_base_jit output
#         "audio": Array2D(shape=(1, 8192), dtype="float32"),  # From expand_dims(wav, 0)
#         "sample_rate": Value(dtype="int64"),
#     }
# )

# lj_speech_data = datasets.load_dataset(
#     "keithito/lj_speech", trust_remote_code=True
# ).with_format("jax")
# lj_speech_data = lj_speech_data.map(
#     transform,
#     # num_proc=8,
#     features=features,
#     remove_columns=lj_speech_data["train"].column_names,  # Remove original columns
# )


# lj_speech_data.save_to_disk("transformed_lj_speech")
lj_speech_data = datasets.load_from_disk("transformed_lj_speech")
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


# pouet_audio = train_data["audio"]
# pouet_mel = train_data["mel"]


# def get_batches(dataset, batch_size):
#     print("h")
#     indices = jnp.arange(len(dataset))
#     # Optionally shuffle indices here

#     for i in range(0, len(dataset), batch_size):
#         batch_idx = indices[i : i + batch_size]
#         yield {"mel": pouet_mel[batch_idx], "audio": pouet_audio[batch_idx]}


# batched_data = train_data.with_format("jax").iter(batch_size=BATCH_SIZE)
step = 0
# Timing stats dictionary
timing_stats = defaultdict(list)

for epoch in range(starting_epoch, N_EPOCHS):
    epoch_start = time.time()
    wait_start = time.time()
    # print("wtf")
    # RANDOM, k = jax.random.split(RANDOM)
    # permutation = jax.random.permutation(k, jnp.arange(0, pouet_audio.shape[0]))
    for i, batch in enumerate(train_data.iter(batch_size=BATCH_SIZE)):

        wait_time = time.time() - wait_start

        # Measure data loading time
        data_load_start = time.time()
        # mels = pouet_mel[permutation[i : i + BATCH_SIZE]]
        # wavs = pouet_audio[permutation[i : i + BATCH_SIZE]]
        mels, wavs = batch["mel"], batch["audio"]
        data_load_time = time.time() - data_load_start

        # Measure training step time
        train_start = time.time()
        results = make_step(
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
        # Force sync for accurate timing
        jax.block_until_ready(results)
        train_time = time.time() - train_start

        # Measure logging time
        log_start = time.time()
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
        ) = results

        step += 1
        writer.add_scalar("Loss/Generator", gan_loss, step)
        writer.add_scalar("Loss/Multi Period", period_loss, step)
        writer.add_scalar("Loss/Multi Scale", scale_loss, step)

        if step % 5 == 0:
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
        log_time = time.time() - log_start

        # Store timing stats
        timing_stats["data_loading"].append(data_load_time)
        timing_stats["training"].append(train_time)
        timing_stats["logging"].append(log_time)
        timing_stats["wait_time"].append(wait_time)

        # Print running averages every 10 steps
        # if i % 1 == 0:
        print(f"\nTiming stats (last 10 steps):")
        print(f"Data loading: {np.mean(timing_stats['data_loading'][-1:]):.3f}s")
        print(f"Training: {np.mean(timing_stats['training'][-1:]):.3f}s")
        print(f"Logging: {np.mean(timing_stats['logging'][-1:]):.3f}s")
        print(f"Iter wait: {np.mean(timing_stats['wait_time'][-1:]):.3f}s")

        wait_start = time.time()

    # Save model
    eqx.tree_serialise_leaves("./generator.eqx", generator)
