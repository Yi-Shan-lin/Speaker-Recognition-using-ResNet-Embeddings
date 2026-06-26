from dataclasses import dataclass, field
from datasets import load_dataset, Features, Array3D, load_from_disk
import torchvision.transforms.v2 as TV
from torchvision.transforms.functional import crop
import torchaudio.transforms as TA
from torchaudio.functional import resample
import torch
from typing import Any, Tuple

"""
Usage example:

from dataset import Dataset_Builder

builder = Dataset_Builder()
builder.filter()
builder.preprocess()
dataset = builder.dataset
sample = dataset.take(32)
"""

@dataclass
class Dataset_Builder:

    # General hyperparameters
    shortest_duration: float = 5.0 # Minimum audio duration in seconds
    longest_duration: float = 33.2 # Maximum audio duration in seconds
    bs: int = 32 # Batch size for mapping
    buffer_size: int = 1000 # How shuffled dataset should be when streaming
    dataset_path: str = "data/dev_dataset" # Path to dataset

    # Audio hyperparameters
    n_fft: int = 400
    hop_len: int = 160
    n_mels: int = 80

    # Vision hyperparameters
    timesteps: int = 400 # Crop size for temporal dimension of log-mel spectograms
    dbfs_min: int = -100 # Min volume
    dbfs_max: int = 40 # Max volume
    mean: Tuple[float] = (0.485, 0.456, 0.406)# [0.569, 0.569, 0.569] (Approximate VoxCeleb2 mean)
    std: Tuple[float] = (0.229, 0.224, 0.225) # [0.110, 0.110 , 0.110] (Approximate VoxCeleb2 std)

    # Data
    dataset: Any = field(init=False)
    features: Any = field(init=False)
    trans: Tuple[Any, Any] = field(init=False)

    def __post_init__(self):
        self.dataset = load_from_disk(self.dataset_path)
        self.features = self.dataset.features

        input_duration = self.timesteps * self.hop_len / 16000
        print("Duration of cropped log-mel spectograms (input): ", input_duration, " seconds")

        if input_duration > self.shortest_duration:
            raise ValueError(f"WARNING: Training might crash: {input_duration} (input duration) > {self.shortest_duration} (shortest duration).")

        self.trans = self._get_trans()

        
    def filter(self):
        # 1. 99.3% audio samples shorter than 33.2s (filter outliers)
        # 2. filter out audio samples shorter than self.shortest_duration
        def pass_criteria(example):            
            audio = example["audio_path"]
            samples = audio.get_all_samples()
            return self.shortest_duration < samples.duration_seconds < self.longest_duration
        
        self.dataset = self.dataset.filter(lambda example: pass_criteria(example))

    def preprocess(self):
        audio_trans, vision_trans = self.trans

        def transform(examples):
            log_mels = []

            for audio in examples["audio_path"]:
                samples = audio.get_all_samples()

                log_mel = audio_trans(samples.data)

                _, h, w = log_mel.shape

                # No center crop like ImageNet, but random crop along temporal dimension of constant length
                left = torch.randint(low=0, high=w-self.timesteps, size=(1,)).item()
                log_mel = crop(img=log_mel, top=0, left=left, height=h, width=self.timesteps)

                # Scale to [0, 1] like for ImageNet
                log_mel = torch.clamp(input=log_mel, min=self.dbfs_min, max=self.dbfs_max) # Reasonable range
                log_mel = (log_mel - self.dbfs_min) / (self.dbfs_max - self.dbfs_min)

                # 3 channels like ImageNet data
                log_mel = log_mel.repeat(3, 1, 1)

                log_mel = vision_trans(log_mel)
                log_mels.append(log_mel)


            examples["log_mel"] = [mel.numpy() for mel in log_mels]
            return examples
        
        # Add new features
        new_features = Features({
            "log_mel": Array3D(shape=(3, 224,224), dtype="float32")
        })
        self.features = Features({**self.features, **new_features})

        # Transform audio and resulting log-mel spectograms
        self.dataset = self.dataset.map(transform, batched=True, batch_size=self.bs, features=self.features)

        # Just in case
        self.dataset = self.dataset.with_format("torch")

    def _get_trans(self):
        audio_transforms = torch.nn.Sequential(
            TA.MelSpectrogram(sample_rate=16000, n_fft=self.n_fft, hop_length=self.hop_len, n_mels=self.n_mels),
            TA.AmplitudeToDB(),
            )
        
        vision_transforms = TV.Compose([
            TV.Resize((224,224)), # ImageNet size
            TV.Normalize(mean=self.mean, std=self.std)
        ])

        return audio_transforms, vision_transforms
        
