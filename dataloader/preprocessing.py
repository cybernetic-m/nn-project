import torch
import torchaudio
import random
import torchaudio.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn


def Time_Stretching(waveform, low, high):
    time_stretch_rate = random.uniform(low, high)  # Time stretching factor
    stretch_transform = transforms.TimeStretch(fixed_rate=time_stretch_rate)
    waveform = stretch_transform(waveform) 
    waveform = torch.clamp(waveform, -1.0, 1.0)
    
    return waveform

def Pitch_Shifting(waveform, sample_rate, low, high):
    pitch_shift_steps = random.randint(low, high)     # Pitch shift steps in semitones
    pitch_shift_transform = transforms.PitchShift(sample_rate, pitch_shift_steps)
    waveform = pitch_shift_transform(waveform) 
    waveform = torch.clamp(waveform, -1.0, 1.0)
    
    return waveform

class Additive_Noise(nn.Module):

    def __init__(self, device):
        super().__init__()

        self.device = device

    def forward(self, waveform):
        noise = torch.randn_like(waveform)
        add_noise = torchaudio.transforms.AddNoise()
        snr_value= 10
        snr = torch.tensor([snr_value], device=self.device)
        white_noise_audio = add_noise(waveform, noise, snr)
        
        return white_noise_audio

def Gain_Adjustment(waveform, gain):
    gain_db = random.uniform(-gain, gain)  
    gain_transform = transforms.Vol(gain_db)
    waveform = gain_transform(waveform) 
    waveform = torch.clamp(waveform, -1.0, 1.0)
    
    return waveform

def SpecAugmentTime(waveform, sample_rate, time_mask_param=30):
    n_fft = 512
    n_mels = 40
    hop_length = n_fft // 2

    # Create MelSpectrogram
    spec_transform = transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length)

    # Generate mel spectrogram
    mel_spectrogram = spec_transform(waveform)

    # Apply time masking
    time_mask = transforms.TimeMasking(time_mask_param)
    mel_spectrogram = time_mask(mel_spectrogram)

    # Convert MelSpectrogram back to linear spectrogram
    inverse_mel_transform = transforms.InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels)
    linear_spectrogram = inverse_mel_transform(mel_spectrogram)

    # No padding here; the frequency dimension should already be n_fft // 2 + 1 (257 for n_fft=512)

    # Apply Griffin-Lim
    window = torch.hann_window(n_fft)
    waveform = torchaudio.functional.griffinlim(linear_spectrogram, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window,
                                                power=2.0, n_iter=16, momentum=0.99, length=waveform.size(-1), rand_init=True)

    # In-place clamping to reduce memory allocation
    waveform.clamp_(-1.0, 1.0)

    return waveform

def SpecAugmentFreq(waveform, sample_rate, freq_mask_param=30):
    n_fft = 512
    n_mels = 40
    hop_length = n_fft // 2

    # Create MelSpectrogram
    spec_transform = transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length)

    # Generate mel spectrogram
    mel_spectrogram = spec_transform(waveform)

    # Apply frequency masking
    freq_mask = transforms.FrequencyMasking(freq_mask_param)
    mel_spectrogram = freq_mask(mel_spectrogram)

    # Convert MelSpectrogram back to linear spectrogram
    inverse_mel_transform = transforms.InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels)
    linear_spectrogram = inverse_mel_transform(mel_spectrogram)

    # No padding here; the frequency dimension should already be n_fft // 2 + 1 (257 for n_fft=512)

    # Apply Griffin-Lim
    window = torch.hann_window(n_fft)
    waveform = torchaudio.functional.griffinlim(linear_spectrogram, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window,
                                                power=2.0, n_iter=16, momentum=0.99, length=waveform.size(-1), rand_init=True)

    # In-place clamping to reduce memory allocation
    waveform.clamp_(-1.0, 1.0)

    return waveform

def Reverberation(waveform):
    reverb_transform = transforms.Reverberate()
    waveform = reverb_transform(waveform) 
    waveform = torch.clamp(waveform, -1.0, 1.0)
    
    return waveform    

def Equalization(waveform, sample_rate):
    eq_transform = transforms.Equalizer(sample_rate)
    waveform = eq_transform(waveform) 
    waveform = torch.clamp(waveform, -1.0, 1.0)
    
    return waveform    

def Resampling(waveform, sample_rate, frequency = [8000, 16000, 22050]):
    resample_transform = transforms.Resample(orig_freq=sample_rate, new_freq=random.choice(frequency))
    waveform = resample_transform(waveform)
 
    waveform = torch.clamp(waveform, -1.0, 1.0)
    
    return waveform

def invert_audio(waveform):
    return waveform.flip(dims=[1])

def flip_waveform(waveform):
    return -waveform

def speed_perturbation(waveform, sample_rate, speed_factor=1.1):
    new_sample_rate = int(sample_rate * speed_factor)
    resample_transform = transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    return resample_transform(waveform)

def pitch_and_speed_perturbation(waveform, sample_rate, speed_factor=1.1, pitch_shift_steps=2):

    waveform = speed_perturbation(waveform, sample_rate, speed_factor)
    
    waveform = Pitch_Shifting(waveform, sample_rate, -pitch_shift_steps, pitch_shift_steps)
    
    return waveform

def mfcc(waveform, sample_rate, mean_signal_length, embedding_length):
    if waveform.shape[1] < mean_signal_length:
        len_padding = mean_signal_length - waveform.shape[1]
        rem_padding = len_padding
        len_padding //= 2
        waveform = F.pad(waveform, (len_padding+rem_padding, len_padding), value=0)
    else:
        len_padding = mean_signal_length - waveform.shape[1]
        len_padding //= 2
        waveform = waveform[len_padding:len_padding+mean_signal_length]
    transforms = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=embedding_length)
    return transforms(waveform)

def mel_spectogram(waveform, sample_rate, mean_signal_length, n_filters):
    if waveform.shape[1] < mean_signal_length:
        len_padding = mean_signal_length - waveform.shape[1]
        rem_padding = len_padding
        len_padding //= 2
        waveform = F.pad(waveform, (len_padding+rem_padding, len_padding), value=0)
    else:
        len_padding = mean_signal_length - waveform.shape[1]
        len_padding //= 2
        waveform = waveform[len_padding:len_padding+mean_signal_length]
    transforms = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_filters)
    return transforms(waveform)
