# HiFiGAN-for-JAX

[![Project Status: Working](https://img.shields.io/badge/status-working-brightgreen.svg)](https://github.com/yourusername/Audio-VQVAE-for-JAX)

A JAX-based implementation of HiFiGAN, heavily based on XTTS. This model is part of the JAXTTS (eXtended Text-To-Speech) series, where I rewrite XTTS in JAX to understand how it works from A to Z, and learn JAX along the way.

## Overview

This project leverages **JAX** and **Equinox** for a HiFiGAN model focused on audio data, copying [XTTS](https://github.com/coqui-ai/TTS)'s structure.

## Features

- A generator mapping Mel Spectrograms to Audio, and a variant used by XTTS mapping GPT2 latents learned by a VQVAE to Audio, conditionned on a ResNet34.
- Comprehensive JAX and Equinox integration
- Documentation with step-by-step tutorials and explanations for each module
- A notebook with code to map the weights of the Coqui-ai's to the JAX implementation. ConvTranspose1d too !

## Getting Started

To get started, clone the repository and follow the tutorial on https://tugdual.fr/HiFiGAN-for-JAX/