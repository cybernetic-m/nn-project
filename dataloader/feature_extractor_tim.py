import torch
import torchaudio.transforms as T

class feature_extractor():
    def __init__(self, sample_rate = 22050, frame_length_ms = 50, frame_shift_ms = 12.5, n_ftt=2048, n_mfcc=39):
       
        # Convert frame length and shift from milliseconds to samples
        frame_length = int(sample_rate * (frame_length_ms / 1000))  # 50 ms to samples
        frame_shift = int(sample_rate * (frame_shift_ms / 1000))  # 12.5 ms to samples

        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate, # Sampling rate (22.05 kHz)
            n_mfcc=n_mfcc, # Number of MFCCs to extract
            melkwargs={
                "n_fft": n_ftt, # FFT points
                "n_mels": 128, # Number of Mel Filter Banks
                "hop_length": frame_shift, # Shift in samples
                "win_length": frame_length, # Length in samples
                "window_fn": torch.hamming_window # Hamming Window
            }
        )

    def apply(self, waveform):

        waveform = waveform.cpu()

        mfcc = self.mfcc_transform(waveform)
        # Average features across channels
        #mfcc_mean = mfcc.mean(dim=0).mean(dim=1)  # Average across channels

        print(mfcc.shape)

        return mfcc
    
   