from dataclasses import dataclass, field
from datasets import load_dataset, Features, Array3D
import torchvision.transforms.v2 as TV
from torchvision.transforms.functional import crop
import torchaudio.transforms as TA
from torchaudio.functional import resample
import torch
import pandas as pd
from typing import Any, Dict, Tuple

"""
Usage example:

from dataset import Dataset_Builder

builder = Dataset_Builder()
builder.filter()
builder.preprocess()
dataset = builder.dataset
sample = dataset.take(32)

Issues:

Runtime too long when using following lines

if example["speaker_id"] not in self.speakers:
    return False

or

self.dataset.shuffle(buffer_size=self.buffer_size, seed=self.seed)
"""

@dataclass
class Dataset_Builder:

    # General hyperparameters
    shortest_duration: float = 5.0 # Minimum audio duration in seconds
    longest_duration: float = 33.2 # Maximum audio duration in seconds
    num_speakers: int = 50 # Number of unique speakers to include in dataset
    bs: int = 32 # Batch size for mapping
    seed: int = 42 # Seed for random operations
    buffer_size: int = 1000 # How shuffled dataset should be when streaming

    # Audio hyperparameters
    target_sample_rate: int = 16000 # Input sample rate in Hz
    n_fft: int = 400
    hop_len: int = 160
    n_mels: int = 80

    # Vision hyperparameters
    timesteps: int = 400 # Crop size for temporal dimension of log-mel spectograms
    dbfs_min: int = -100 # Min volume
    dbfs_max: int = 40 # Max volume
    mean: float = 0.569 # Approximate VoxCeleb2 mean
    std: float = 0.110 # Approximate VoxCeleb2 std

    # Data
    dataset: Any = field(default_factory=lambda: load_dataset("acul3/voxceleb2", split="train", streaming=True))
    features: Any = field(init=False)
    speakers: Dict[str, int] = field(init=False)
    trans: Tuple[Any, Any] = field(init=False)

    def __post_init__(self):
        self.features = self.dataset.features

        input_duration = self.timesteps * self.hop_len / self.target_sample_rate
        print("Duration of cropped log-mel spectograms (input): ", input_duration, " seconds")

        if input_duration > self.shortest_duration:
            raise ValueError(f"WARNING: Training might crash: {input_duration} (input duration) > {self.shortest_duration} (shortest duration).")
        
        df = pd.read_csv("data/speaker_count.csv", nrows=self.num_speakers)
        self.speakers = df.set_index("speaker_id")["count"].to_dict()

        self.trans = self._get_trans()

        
    def filter(self):
        # 99.3% english (filter outliers)
        self.dataset = self.dataset.filter(lambda example: example["language"] == "en")

        # Only require audio and speaker ID
        self.dataset = self.dataset.remove_columns(["transcription", "language", "gender"])
        self.features = self.dataset.features

        # 1. Only keep n most represented speakers (num_speakers)
        # 2. 99.3% audio samples shorter than 33.2s (filter outliers)
        # 3. filter out audio samples shorter than self.shortest_duration
        def pass_criteria(example):
            #if example["speaker_id"] not in self.speakers:
            #    return False
            
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

                waveform = resample(waveform=samples.data, orig_freq=samples.sample_rate, new_freq=self.target_sample_rate)
                log_mel = audio_trans(waveform)

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


            examples["log_mel"] = torch.stack(log_mels)
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

    def get_dataset(self):
        return self.dataset.shuffle(buffer_size=self.buffer_size, seed=self.seed)

    def _get_trans(self):
        audio_transforms = torch.nn.Sequential(
            TA.MelSpectrogram(sample_rate=self.target_sample_rate, n_fft=self.n_fft, hop_length=self.hop_len, n_mels=self.n_mels),
            TA.AmplitudeToDB(),
            )
        
        vision_transforms = TV.Compose([
            TV.Resize((224,224)), # ImageNet size
            TV.Normalize(mean=[self.mean, self.mean, self.mean], std=[self.std, self.std, self.std]) # Since images are out of distribution, use approximate dataset statistics
        ])

        return audio_transforms, vision_transforms
        
