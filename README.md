# HiFiGAN-for-JAX

[![Project Status: Working](https://img.shields.io/badge/status-working-brightgreen.svg)](https://tugdual.fr/HiFiGAN-for-JAX/)

A JAX-based implementation of HiFiGAN, heavily based on [the original code](https://github.com/jik876/hifi-gan). This model is part of the JAXTTS (eXtended Text-To-Speech) series, where I rewrite XTTS in JAX to understand how it works from A to Z, and learn JAX along the way.

## Overview

This project leverages **JAX** and **Equinox** for a HiFiGAN model focused on audio data.

## 🚗 Roadmap

- [x] Functioning vocoder mapping Mel spec to Audio
- [x] JIT accelerated Mel transform
- [x] Documentation with step-by-step tutorials and explanations for each module
- [x] A notebook with code to map the weights of the Coqui-ai's to the JAX implementation. ConvTranspose1d too !
- [x] Accelerate the dataloading. Changing the HF datasets' features from Sequence(Sequence(Float)) to Array2D improved training time by x16 !
- [ ] More complete profiling of the model
- [ ] Validation loss
- [ ] Model checkpoints available
- [ ] Full type annotation
- [ ] DocStrings on all functions
- [ ] Speed comparison to original HiFiGAN


## Getting Started

To get started, follow the commands below. I recommend you use [UV](https://docs.astral.sh/uv/) as a package manager: 

```bash
git clone git@github.com:TugdualKerjan/hifigan-jax.git
cd hifigan-jax
uv sync
uv add jax["cuda"] # JAX has various versions optimized for the underlying architecture
uv run HiFiGAN.py
```

## Things I've learned with this project

- How to use the uv package manager
- Basic PyTesting
- Ruff as a code formatter
- Beartyping basics with Jaxtyping to avoid tensor shape errors
- The dataloading can be a huge area of improvement for speed.
- Caching JAX Jitted code to save compilation time every re-run
- Using from `config.update("jax_debug_nans", True)` to figure out that `, precision="highest")` in a `jnp.matmul` avoids nans
- Memory and compute profiling using various techniques, mainly [pprof](https://github.com/google/pprof) and [tensorboard profiler](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras)
- Understanding why the code was slow because of recompilations linked to varying shape inputs in jitted functions. Used below code to find them:

```python
from jax import config
import logging

# Enable both compilation logging and shape checking
config.update("jax_log_compiles", True)  # Basic logging

# Custom filter to only show shapes
class ShapeFilter(logging.Filter):
    def filter(self, record):
        return "Compiling" in record.msg and "shapes and types" in record.msg


# Set up logger for shapes
shape_logger = logging.getLogger("jax._src.interpreters.pxla")
shape_logger.addFilter(ShapeFilter())
shape_logger.setLevel(logging.WARNING)
```
