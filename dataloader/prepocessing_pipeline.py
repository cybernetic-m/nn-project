import torch.nn as nn
from preprocessing import pitch_and_speed_perturbation, speed_perturbation, SpecAugmentFreq, SpecAugmentTime

class preprocessing_pipeline(nn.Module):
    def __init__(self, speed_factor, pitch_shift_step, freq_mask_param, time_mask_param, mean_signal_length):
        super().__init__()
        self.speed_factor = speed_factor
        self.pitch_shift_step = pitch_shift_step
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.mean_signal_length = mean_signal_length

    def forward(self, waveform, sample_rate):
        waveform1 = pitch_and_speed_perturbation(waveform, sample_rate, self.speed_factor, self.pitch_shift_step)
        waveform2 = speed_perturbation(waveform, sample_rate, self.speed_factor)
        waveform3 = SpecAugmentFreq(waveform, sample_rate, self.freq_mask_param)
        waveform4 = SpecAugmentTime(waveform, sample_rate, self.time_mask_param)
       
        return waveform1, waveform2, waveform3, waveform4