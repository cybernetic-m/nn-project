from torch.utils.data import Dataset
import os
import torchaudio
import torch

class EMOVO_Dataset(Dataset):

  def __init__(self, dataset_dir, transform=None, device = 'cpu'):

    self.dataset_dir = dataset_dir
    self.transform = transform
    self.classes = ['dis', 'gio', 'neu', 'pau', 'rab', 'sor', 'tri']
    self.data = [] # Waw Audio as tensors
    self.labels = [] # Labels of the data

    file_list = os.listdir(dataset_dir)
    for filename in file_list:
      path = os.path.join(dataset_dir, filename)
      tensor, sample_rate = torchaudio.load(path)
      if tensor.shape[0] == 1: # if an audio is mono we duplicate the channel to make it stereo 
        tensor = torch.cat((tensor,tensor))
        print("fixed a mono track:",filename)
        torchaudio.save(path, tensor, sample_rate)
      class_ = self.classes.index(filename.split('-')[0])
      self.data.append((tensor,sample_rate))
      self.labels.append(torch.tensor(class_))
    self.device = device

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):

    data = self.data[idx]
    data = (data[0].to(self.device), data[1])
    label = self.labels[idx].to(self.device)

    return data, label

  def get_info(self):
    n_sample = [ self.labels.count(class_) for class_ in range(len(self.classes))]
    return n_sample