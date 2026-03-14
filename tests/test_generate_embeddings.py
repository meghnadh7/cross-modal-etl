import numpy as np
import torch
from torch.utils.data import Dataset

from cross_modal.generate_embeddings import generate_audio_embeddings, generate_visual_embeddings


class DummyVisualDataset(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return {
            "image": torch.ones(3, 224, 224) * (idx + 1),
            "caption": f"caption-{idx}",
            "id": f"img-{idx}",
            "valid": idx != 1,
        }


class DummyAudioDataset(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return {
            "audio": torch.ones(1, 16000) * (idx + 1),
            "caption": f"audio-{idx}",
            "id": f"aud-{idx}",
            "valid": idx != 2,
        }


class FakeClipEngine:
    def encode_image_tensors(self, images):
        return np.ones((images.shape[0], 8), dtype=np.float32)

    def encode_texts(self, texts):
        return np.zeros((len(list(texts)), 8), dtype=np.float32)


class FakeClapEngine:
    def encode_audio_tensors(self, audios):
        return np.ones((audios.shape[0], 6), dtype=np.float32)

    def encode_texts(self, texts):
        return np.zeros((len(list(texts)), 6), dtype=np.float32)


def test_generate_visual_embeddings_skip_invalid():
    dataset = DummyVisualDataset()
    result = generate_visual_embeddings(
        dataset,
        clip_engine=FakeClipEngine(),
        batch_size=2,
        num_workers=0,
        skip_invalid=True,
    )

    assert result["image_embeddings"].shape == (2, 8)
    assert result["text_embeddings"].shape == (2, 8)
    assert len(result["metadata"]) == 2
    assert all(item["id"] in {"img-0", "img-2"} for item in result["metadata"])


def test_generate_audio_embeddings_keep_invalid():
    dataset = DummyAudioDataset()
    result = generate_audio_embeddings(
        dataset,
        clap_engine=FakeClapEngine(),
        batch_size=2,
        num_workers=0,
        skip_invalid=False,
    )

    assert result["audio_embeddings"].shape == (3, 6)
    assert result["text_embeddings"].shape == (3, 6)
    assert len(result["metadata"]) == 3
