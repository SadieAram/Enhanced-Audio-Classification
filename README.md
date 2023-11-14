# Enhanced-Audio-Classification
Audio classification using Wav2Vec 2.0 and Hugging Face Transformers

This repository implements a Wav2Vec 2.0 model with a Classification-Head using TensorFlow for keyword spotting (KWS) tasks on the Google Speech Commands dataset.

## Table of Contents

1. [Setup](#setup)
2. [Configuration](#configuration)
3. [Data Processing](#data-processing)
4. [Feature Extraction](#feature-extraction)
5. [Model Definition](#model-definition)
6. [Model Training](#model-training)
7. [Model Evaluation](#model-evaluation)
8. [Inference](#inference)
9. [Model Sharing](#model-sharing)

## Setup

### Installing Dependencies

```bash
pip install git+https://github.com/huggingface/transformers.git
pip install datasets
pip install huggingface-hub
pip install joblib
pip install librosa
## Configuration

Define variables such as:

- `MAX_DURATION`: Maximum duration of input audio files (in seconds).
- `SAMPLING_RATE`: Number of samples of audio recorded every second.
- ...

## Data Processing

### Load the Google Speech Commands Dataset

```python
from datasets import load_dataset

speech_commands_v1 = load_dataset("superb", "ks")
