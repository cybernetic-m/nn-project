import torch.nn as nn
from preprocessing import mel_spectogram, mfcc

class spectogram_pipeline(nn.Module):
    def __init__(self, mean_signal_length, embedding_length, log_DCT) -> None:
        super().__init__()
        self.log_DCT = log_DCT
        self.mean_signal_length = mean_signal_length
        self.embedding_length = embedding_length

    def forward(self, waveform, sample_rate):

        if self.log_DCT:
            out = mfcc(waveform, sample_rate, self.mean_signal_length, self.embedding_length)
        else:
            out = mel_spectogram(waveform, sample_rate, self.mean_signal_length, self.embedding_length)
       
        return out