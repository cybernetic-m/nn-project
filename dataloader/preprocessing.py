import torch
import random
import torchaudio.transforms as transforms
import torch.nn as nn
import numpy as np
import librosa

class Preprocessing(nn.Module):

    def __init__(self, device):
        super().__init__()

        self.device = device

    def forward(self, waveform):

        #--------#
        # Start white noise preprocessing
        noise = torch.randn_like(waveform)
        add_noise = transforms.AddNoise()
        snr_value= 10
        snr = torch.tensor([snr_value], device=self.device)
        white_noise_audio = add_noise(waveform, noise, snr)
        # Finish white noise preprocessing
        #--------#
        
        #--------#
        # Start shifting preprocessing
        channels_dim = waveform.shape[0]
        samples_dim = waveform.shape[1]
        random_percentage = random.choice([0.05, 0.1, 0.15, 0.2, 0.3, -0.05, -0.1, -0.15, -0.2, -0.3])
        shift_amount = int(random_percentage * samples_dim)

        zeros_tensor = torch.zeros((channels_dim, abs(shift_amount))).to(self.device)
        
        # Add zeros at the start of the waveform
        if shift_amount > 0:
            shifted_waveform = torch.cat((zeros_tensor, waveform[:, :-shift_amount]), dim=1)

        # Add zeros at the end of the waveform
        else:
            shift_amount = abs(shift_amount)
            shifted_waveform = torch.cat((waveform[:, shift_amount:], zeros_tensor), dim=1)
        # Finish shifting preprocessing
        #--------#
    

        #--------#
        # Start pitching preprocessing    
        speed_factor = random.choice([0.25, 0.5, 1.5, 2])
        waveformcpu = waveform.cpu()
        waveformnumpy = waveformcpu.numpy()
        waveform_pitched = librosa.effects.time_stretch(waveformnumpy, rate=speed_factor)
        waveform_pitched_tensor = torch.tensor(waveform_pitched, device=self.device)
        # Finish pitching preprocessing
        #--------#
        
        #--------#
        # Start reverse preprocessing 
        reversed_audio = torch.flip(waveform, dims=[1]) 
        # Finish reverse preprocessing 
        #--------#
        
        out = (white_noise_audio, shifted_waveform, waveform_pitched_tensor, reversed_audio)

        return out
        
        
def invert_audio(waveform):
    flip_waveform = waveform.flip(dims=[2])
    return flip_waveform


