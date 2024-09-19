import torch
import torchaudio
import random
import torchaudio.transforms as transforms

class preprocessing():
    def __init__(self, dataset) -> None:
        super.__init__(preprocessing)

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

def Additive_Noise(waveform):
    noise_factor = 0.005                          # Noise level
    noise = noise_factor * torch.randn_like(waveform)
    waveform = waveform + noise
    waveform = torch.clamp(waveform, -1.0, 1.0)
    
    return waveform

def Gain_Adjustment(waveform, gain):
    gain_db = random.uniform(-gain, gain)  
    gain_transform = transforms.Vol(gain_db)
    waveform = gain_transform(waveform) 
    waveform = torch.clamp(waveform, -1.0, 1.0)
    
    return waveform

def SpecAugmentTime(waveform, sample_rate, time_mask_param = 30 ):
    spec_transform = transforms.MelSpectrogram(sample_rate=sample_rate)
    mel_spectrogram = spec_transform(waveform)
    time_mask = transforms.TimeMasking(time_mask_param)
    mel_spectrogram = time_mask(mel_spectrogram)
    waveform = torchaudio.functional.griffinlim(mel_spectrogram) 
    waveform = torch.clamp(waveform, -1.0, 1.0)
    
    return waveform

def SpecAugmentFreq(waveform, sample_rate, freq_mask_param = 30):
    spec_transform = transforms.MelSpectrogram(sample_rate=sample_rate)
    mel_spectrogram = spec_transform(waveform)
    freq_mask = transforms.FrequencyMasking(freq_mask_param)
    mel_spectrogram = freq_mask(mel_spectrogram)
    waveform = torchaudio.functional.griffinlim(mel_spectrogram) 
    waveform = torch.clamp(waveform, -1.0, 1.0)
    
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