import torch
import torchaudio.transforms as T
import torch.nn.functional as F

class feature_extractor():
    def __init__(self, waveform_sec = 5000, sample_rate = 22050, frame_length_ms = 50, frame_shift_ms = 12.5, n_ftt=2048, n_mfcc=39):
        
        # Convert waveform_sec from ms to samples
        self.waveform_win = int(48000 * (waveform_sec / 1000)) # 5000 ms to samples

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

        
        # This function firstly add (zeros) or remove samples from the waveform to have the same length
        # After compute the MFCC and return it

        # Padding case (Add zeros): waveform.shape = 50, waveform_win = 325
        if waveform.shape[1] < self.waveform_win:
            # Compute how much zeros to add symmetrically to the signal
            pad_len = self.waveform_win - waveform.shape[1] # pad_len = 275
            pad_remainder = pad_len % 2  # pad_remainder = 1 (Odd Number)
            pad_len = pad_len // 2  # pad_len = 137 (Integer part of the division)

            # Add zeros
            # 137 to the left and 138 to the right part
            # padded_waveform = [0,0,0 ... waveform ... 0 0 0]
            # padded_waveform.shape = 137 (left) + 50 (waveform) + 138 (right) = 325 = waveform_win
            padded_waveform = F.pad(waveform, [pad_len, pad_len + pad_remainder], "constant", value=0) 

            padded_waveform = padded_waveform.cpu()

            # MFCC computation
            mfcc = self.mfcc_transform(padded_waveform)
            
            # Flattening
            mfcc = mfcc.reshape(mfcc.shape[0], -1) # mfcc.shape = [2, 34047 (39*873)]

            

        # Trimming Case (Remove samples): waveform.shape = 200, waveform_win = 100                         
        else:
            # Compute how much zeros to remove symmetrically to the signal
            pad_len = waveform.shape[1] - self.waveform_win # pad_len = 100
            pad_len = pad_len // 2 # pad_len = 50

            # Take the centered part
            trimmed_waveform =  waveform[:,pad_len : pad_len+self.waveform_win] # trimmed_waveform[50:150] => trimmed_waveform.shape = 100

            trimmed_waveform = trimmed_waveform.cpu()

            # MFCC computation
            mfcc = self.mfcc_transform(trimmed_waveform) # mfcc.shape = [2, 39, 873]

            # Flattening
            mfcc = mfcc.reshape(mfcc.shape[0], -1) # mfcc.shape = [2, 34047 (39*873)]

        return mfcc
    
   