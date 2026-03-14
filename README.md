# Cross-Modal ETL Pipeline for Semantic Audio-Visual Retrieval

This repository now includes the **preprocessing + embedding generation stage** for a cross-modal retrieval system:

- **Images**: COCO-style image/caption data -> CLIP embeddings
- **Audio**: AudioCaps/ESC-like captioned audio -> CLAP embeddings
- **Text pivot**: caption text embeddings generated alongside each modality

## What is implemented

### 1) Ingestion + preprocessing (`cross_modal/ingestion.py`)

- `VisualDataset`
  - Reads COCO-style annotations (`annotations` list with `image_id`, `caption`)
  - Resizes images and applies CLIP normalization
  - Handles missing/corrupt files with safe fallback tensors

- `AudioDataset`
  - Loads captioned audio datasets from Hugging Face (`TwinkStart/AudioCaps` by default)
  - Converts to mono, resamples, and pads/truncates to fixed duration
  - Filters silent/low-quality audio using RMS threshold

### 2) Embedding engine (`cross_modal/embeddings.py`)

- `CLIPEmbeddingEngine`
  - Image embeddings: `get_image_features`
  - Text embeddings: `get_text_features`

- `CLAPEmbeddingEngine`
  - Audio embeddings: `get_audio_features`
  - Text embeddings: `get_text_features`

All embeddings are L2-normalized for cosine-similarity / ANN workflows.

### 3) End-to-end embedding job (`cross_modal/generate_embeddings.py`)

Runs the ETL transform stage and writes:

- `clip_image_embeddings.npy`
- `clip_text_from_image_captions.npy`
- `clap_audio_embeddings.npy`
- `clap_text_from_audio_captions.npy`
- `image_metadata.jsonl`
- `audio_metadata.jsonl`
- `run_config.json`

## Install

```bash
pip install -r requirements.txt
```

## Example run

```bash
python -m cross_modal.generate_embeddings \
  --output-dir ./artifacts \
  --image-dir /path/to/coco/images/train2017 \
  --coco-annotations /path/to/coco/annotations/captions_train2017.json \
  --audio-cache-dir /path/to/hf_cache \
  --audio-dataset-name TwinkStart/AudioCaps \
  --audio-split train \
  --batch-size 32 \
  --skip-invalid
```

## Notes

- Use `--image-limit` / `--audio-limit` for small-scale dry runs.
- For GPU throughput, keep FP16 enabled (default). Use `--disable-fp16` if needed.
- If you want ESC-50, set `--audio-dataset-name` to an ESC-50 dataset available on Hugging Face.
