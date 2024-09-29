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

    def forward(self, waveform, sample_rate):

        type_of_prep = random.choice([0, 1, 2, 3])
        
        if type_of_prep == 0:
            noise = torch.randn_like(waveform)
            add_noise = transforms.AddNoise()
            snr_value= 10
            snr = torch.tensor([snr_value], device=self.device)
            white_noise_audio = add_noise(waveform, noise, snr)
            
            return white_noise_audio, type_of_prep
        
        elif type_of_prep == 1:
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
            return shifted_waveform, type_of_prep

        # Do the pitch randomically taking a random number of semitones "n_steps"
        elif type_of_prep == 2:
            delay_time = random.choice([0.6, 1, 2])
            decay_factor = random.choice([0.4, 0.7, 0.9])
            delay_samples = int(delay_time * sample_rate)
            waveformcpu = waveform.cpu()
            delayed_waveform = np.zeros_like(waveformcpu)
            delayed_waveform[delay_samples:] = waveformcpu[:-delay_samples]
            waveform_with_echo = waveformcpu + delayed_waveform * decay_factor
            return waveform_with_echo.to(self.device), type_of_prep
        elif type_of_prep == 3:
            speed_factor = random.choice([0.25, 0.5, 1.5, 2])
            waveformcpu = waveform.cpu()
            waveformnumpy = waveformcpu.numpy()
            waveform_pitched = librosa.effects.time_stretch(waveformnumpy, rate=speed_factor)
            return torch.tensor(waveform_pitched, device=self.device), type_of_prep


def invert_audio(waveform):
    return waveform.flip(dims=[1])


