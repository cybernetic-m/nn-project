
import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
import random
import torch.nn.functional as F

class AddNoise(nn.Module):
    def __init__(self, noise_factor=0.005):
        super(AddNoise, self).__init__()
        self.noise_factor = noise_factor

    def forward(self, mfcc):
        noise = self.noise_factor * torch.randn(mfcc.size())
        return mfcc + noise

class TimeShift(nn.Module):
    def __init__(self, shift_max=2):
        super(TimeShift, self).__init__()
        self.shift_max = shift_max

    def forward(self, mfcc):
        shift = random.randint(-self.shift_max, self.shift_max)
        if shift > 0:
            return torch.cat((mfcc[shift:], torch.zeros(shift, mfcc.size(1))), dim=0)
        elif shift < 0:
            return torch.cat((torch.zeros(-shift, mfcc.size(1)), mfcc[:shift]), dim=0)
        return mfcc

class RandomCrop(nn.Module):
    def __init__(self, target_length):
        super(RandomCrop, self).__init__()
        self.target_length = target_length

    def forward(self, mfcc):
        start = random.randint(0, mfcc.size(0) - self.target_length)
        return mfcc[start:start + self.target_length]

class FrequencyMasking(nn.Module):
    def __init__(self, freq_mask_param=2):
        super(FrequencyMasking, self).__init__()
        self.freq_mask_param = freq_mask_param

    def forward(self, mfcc):
        mfcc_copy = mfcc.clone()
        num_freqs = mfcc.size(1)
        mask_start = random.randint(0, num_freqs - self.freq_mask_param)
        mfcc_copy[:, mask_start:mask_start + self.freq_mask_param] = 0
        return mfcc_copy

class TimeMasking(nn.Module):
    def __init__(self, time_mask_param=2):
        super(TimeMasking, self).__init__()
        self.time_mask_param = time_mask_param

    def forward(self, mfcc):
        mfcc_copy = mfcc.clone()
        num_times = mfcc.size(0)
        mask_start = random.randint(0, num_times - self.time_mask_param)
        mfcc_copy[mask_start:mask_start + self.time_mask_param, :] = 0
        return mfcc_copy

class ShiftMFCC(nn.Module):
    def __init__(self, shift_max=2):
        super(ShiftMFCC, self).__init__()
        self.shift_max = shift_max

    def forward(self, mfcc):
        shift = random.randint(-self.shift_max, self.shift_max)
        if shift > 0:
            return torch.cat((torch.zeros(shift, mfcc.size(1)), mfcc[:-shift]), dim=0)
        elif shift < 0:
            return torch.cat((mfcc[-shift:], torch.zeros(-shift, mfcc.size(1))), dim=0)
        return mfcc

class TimeStretch(nn.Module):
    def __init__(self, stretch_factor):
        super(TimeStretch, self).__init__()
        self.stretch_factor = stretch_factor

    def forward(self, mfcc):
        original_length = mfcc.size(0)
        new_length = int(original_length * self.stretch_factor)
        return F.interpolate(mfcc.unsqueeze(0), size=new_length, mode='linear', align_corners=True).squeeze(0)
