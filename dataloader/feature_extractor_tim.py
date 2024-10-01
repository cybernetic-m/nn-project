import torch
import torchaudio.transforms as T

class feature_extractor():
    def __init__(self, dataset, frame_length = 2400, frame_shift = 600):
        self.dataset = dataset
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.features = []

    def apply(self):

        for data, _ in self.dataset:
            waveform, sample_rate = data
            waveform = waveform.cpu()

            mfcc = self.mfcc(waveform, sample_rate, 39)
            # Average features across channels
            mfcc_mean = mfcc.mean(dim=0).mean(dim=1)  # Average across channels

            self.features.append(mfcc_mean)
        
    def get_features(self):
        return self.features

    def mfcc(self, waveform, sample_rate, n_mfcc):
        n_fft = 2048
        win_length = min(n_fft, self.frame_length)
        mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "n_mels": 128,
                "hop_length": self.frame_shift,
                "win_length": win_length,
                "window_fn": torch.hamming_window
            }
        )
        mfcc = mfcc_transform(waveform)
        return mfcc
    
   