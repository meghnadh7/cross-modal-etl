import os
import json
import torch
import torchaudio
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset


class VisualDataset(Dataset):
    """Data loader for MS COCO images and captions."""

    def __init__(self, image_dir, annotation_file):
        self.image_dir = image_dir

        # Load the COCO JSON file
        with open(annotation_file, 'r') as f:
            data = json.load(f)
            self.annotations = data['annotations']

            # Visual Norm: Resize 224x224 and normalize for CLIP
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_id = str(ann['image_id']).zfill(12)
        caption = ann['caption']

        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")

        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            # Fallback for missing/corrupt images to prevent pipeline crashes
            image = torch.zeros((3, 224, 224))

        return {"image": image, "caption": caption, "id": img_id}


class AudioDataset(Dataset):
    """Data loader for AudioCaps, reading directly from the Hugging Face cache."""

    def __init__(self, cache_dir, split="train", target_sr=44100, duration_sec=10):
        self.target_sr = target_sr
        self.target_length = target_sr * duration_sec

        # Load directly from your external cache folder
        self.hf_dataset = load_dataset(
            "TwinkStart/AudioCaps",
            split=split,
            cache_dir=cache_dir
        )

    def validate_audio(self, waveform):
        """Filter low-quality/silent clips by checking Root Mean Square energy."""
        rms_energy = torch.sqrt(torch.mean(waveform ** 2))
        return rms_energy > 0.001

    def normalize_audio(self, waveform, sr):
        """Audio Norm: Resample 44.1kHz, Mono, Pad/Truncate 10s."""
        # 1. Convert to Mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 2. Resample to 44.1kHz
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        # 3. Pad or Truncate to exactly 10 seconds
        if waveform.shape[1] > self.target_length:
            waveform = waveform[:, :self.target_length]
        elif waveform.shape[1] < self.target_length:
            pad_amount = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        return waveform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        row = self.hf_dataset[idx]
        caption = row['caption']

        # Hugging Face automatically decodes the audio into a numpy array for us
        audio_array = row['audio']['array']
        sr = row['audio']['sampling_rate']

        # Convert numpy array to PyTorch tensor
        waveform = torch.tensor(audio_array).float()

        # Ensure it has a channel dimension (1, length)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Validate and Normalize
        if not self.validate_audio(waveform):
            return {"audio": torch.zeros((1, self.target_length)), "caption": caption, "valid": False}

        waveform = self.normalize_audio(waveform, sr)

        return {"audio": waveform, "caption": caption, "valid": True}