import json
import os
from typing import Any, Dict

import torch
import torchaudio
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


class VisualDataset(Dataset):
    """Data loader for image-caption data (COCO-style annotation JSON)."""

    def __init__(self, image_dir: str, annotation_file: str, image_size: int = 224):
        self.image_dir = image_dir
        self.image_size = image_size

        with open(annotation_file, "r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
        self.annotations = data.get("annotations", [])

        # CLIP-style visual normalization.
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(CLIP_IMAGE_MEAN, CLIP_IMAGE_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ann = self.annotations[idx]
        img_id = str(ann.get("image_id", idx)).zfill(12)
        caption = ann.get("caption", "")
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")

        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            valid = True
        except Exception:
            # Fallback for missing/corrupt images to avoid crashing long ETL jobs.
            image = torch.zeros((3, self.image_size, self.image_size))
            valid = False

        return {"image": image, "caption": caption, "id": img_id, "valid": valid}


class AudioDataset(Dataset):
    """Data loader for captioned audio datasets (AudioCaps by default)."""

    def __init__(
        self,
        cache_dir: str,
        split: str = "train",
        target_sr: int = 44100,
        duration_sec: int = 10,
        dataset_name: str = "TwinkStart/AudioCaps",
    ):
        self.target_sr = target_sr
        self.target_length = target_sr * duration_sec
        self.dataset_name = dataset_name
        self.hf_dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)

    def validate_audio(self, waveform: torch.Tensor) -> bool:
        """Filter low-quality/silent clips by checking RMS energy."""
        rms_energy = torch.sqrt(torch.mean(waveform**2))
        return bool(rms_energy > 0.001)

    def normalize_audio(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Resample, convert to mono, and pad/truncate to fixed length."""
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim > 2:
            waveform = waveform.reshape(waveform.shape[0], -1)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        if waveform.shape[1] > self.target_length:
            waveform = waveform[:, : self.target_length]
        elif waveform.shape[1] < self.target_length:
            pad_amount = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        return waveform

    def _extract_caption(self, row: Dict[str, Any]) -> str:
        if "caption" in row and row["caption"]:
            return row["caption"]
        if "human_labels" in row and row["human_labels"]:
            return str(row["human_labels"])
        if "category" in row and row["category"]:
            return str(row["category"])
        return ""

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.hf_dataset[idx]
        caption = self._extract_caption(row)
        sample_id = str(row.get("audiocap_id", row.get("id", idx)))
        audio_blob = row.get("audio")
        if not audio_blob:
            return {
                "audio": torch.zeros((1, self.target_length)),
                "caption": caption,
                "id": sample_id,
                "valid": False,
                "sampling_rate": self.target_sr,
            }

        audio_array = audio_blob.get("array")
        sr = int(audio_blob.get("sampling_rate", self.target_sr))
        if audio_array is None:
            return {
                "audio": torch.zeros((1, self.target_length)),
                "caption": caption,
                "id": sample_id,
                "valid": False,
                "sampling_rate": sr,
            }

        waveform = torch.tensor(audio_array).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if not self.validate_audio(waveform):
            return {
                "audio": torch.zeros((1, self.target_length)),
                "caption": caption,
                "id": sample_id,
                "valid": False,
                "sampling_rate": sr,
            }

        waveform = self.normalize_audio(waveform, sr)
        return {
            "audio": waveform,
            "caption": caption,
            "id": sample_id,
            "valid": True,
            "sampling_rate": sr,
        }