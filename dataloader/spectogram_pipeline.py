import torch.nn as nn
from preprocessing import mel_spectogram, mfcc

class preprocessing_pipeline(nn.Module):
    def __init__(self, sample_rate, log_DCT) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.log_DCT = log_DCT

    def forward(self, waveform):

        if self.log_DCT:
            out = mfcc(waveform, self.sample_rate, self.mean_signal_length, self.embedding_length)
        else:
            out = mel_spectogram(waveform, self.sample_rate, self.mean_signal_length, self.embedding_length)
       
        return out