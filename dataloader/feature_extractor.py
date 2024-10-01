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

            mel_spectrogram = self.mel_spect(waveform, sample_rate, 128)
            chroma = self.chroma(waveform)
            zcr = self.ZCR(waveform)
            rms = self.RMS(waveform)
            mfcc = self.mfcc(waveform, sample_rate, 40)
            # Average features across channels
            mel_spectrogram_mean = mel_spectrogram.mean(dim=0).mean(dim=1)  # Average across channels
            chroma_mean = chroma.mean(dim=1)  # Average across channels
            zcr_mean = zcr.mean()  # Single value
            rms_mean = rms.mean()  # Single value
            mfcc_mean = mfcc.mean(dim=0).mean(dim=1)  # Average across channels

            # Combine features into a single tensor
            combined_features = torch.cat([
                mel_spectrogram_mean,
                chroma_mean,
                zcr_mean.unsqueeze(0),  # Ensure correct shape
                rms_mean.unsqueeze(0),  # Ensure correct shape
                mfcc_mean
            ], dim=0)
            self.features.append(combined_features)

    def apply_tim(self):
        return ...
        
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
    
    def mel_spect(self, waveform, sample_rate, n_mels):
        n_fft = 2048
        win_length = min(n_fft, self.frame_length)
        mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=self.frame_shift,
        n_mels=n_mels
        )
        mel_spectrogram = mel_spectrogram_transform(waveform)
        return mel_spectrogram
    
    def chroma(self, waveform, n_chroma=12):
        # Compute the short-time Fourier transform (STFT)
        n_fft = 2048
        win_length = min(n_fft, self.frame_length)
        window = torch.hamming_window(win_length, device=waveform.device)
        stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=self.frame_shift,
        win_length=win_length,
        window=window,
        return_complex=True
        )

        # Convert to magnitude
        magnitude = torch.abs(stft).mean(dim=0)
        
        # The shape of magnitude should be (n_freq, n_frames)
        n_freq, n_frames = magnitude.shape
        assert n_freq == 1025, "Expected number of frequency bins should be 1025."
        
        # Initialize chroma features
        chroma = torch.zeros((n_chroma, n_frames), device=waveform.device)
        chroma_bins = torch.tensor([
                0,  # C
                1,  # C#
                3,  # D
                4,  # D#
                5,  # E
                7,  # F
                8,  # F#
                10, # G
                11, # G#
                12, # A
                2,  # A#
                9   # B
            ])

        for i in range(n_chroma):
            # Select the magnitude for the chroma bin, summing across appropriate frequencies
            # The bins are repeated every 12 frequencies.
            chroma[i] = magnitude[chroma_bins[i] + (torch.arange(0, n_freq // 12) * 12).to(waveform.device), :].sum(dim=0)
    
        return chroma

    def ZCR(self, waveform):
        zero_crossings = ((waveform[:, 1:] * waveform[:, :-1]) < 0).sum(dim=1).float()
        return zero_crossings / (waveform.shape[1] - 1)  # Normalize by the number of samples

            
    def RMS(self, waveform):
        rms = torch.sqrt(torch.mean(waveform ** 2, dim=1))
        return rms
    