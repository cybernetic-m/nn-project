from torch.utils.data import Dataset
import os
import torchaudio
import torch
import numpy as np
from dataloader.feature_extractor import feature_extractor

class EMOVO_Dataset(Dataset):

  def __init__(self, dataset_dir, feature_extract=False, mfcc_np = False, device = 'cpu'):

    self.dataset_dir = dataset_dir
    self.classes = ['dis', 'gio', 'neu', 'pau', 'rab', 'sor', 'tri']
    self.data = [] # Waw Audio as tensors
    self.labels = [] # Labels of the data
    
    file_list = os.listdir(dataset_dir)

    if feature_extract:
      extractor = feature_extractor() 
      print("Extracting MFCC features...")
    
    if mfcc_np == False:
      for filename in file_list:
        path = os.path.join(dataset_dir, filename)
        tensor, sample_rate = torchaudio.load(path)

        if tensor.shape[0] == 1: # if an audio is mono we duplicate the channel to make it stereo 
          tensor = torch.cat((tensor,tensor))
          print("fixed a mono track:",filename)
          torchaudio.save(path, tensor, sample_rate)

        if feature_extract:
          features = extractor.apply(tensor)
          class_ = self.classes.index(filename.split('-')[0])
          self.data.append((features,sample_rate))
          self.labels.append(torch.tensor(class_))

        else:
          class_ = self.classes.index(filename.split('-')[0])
          self.data.append((tensor,sample_rate))
          self.labels.append(torch.tensor(class_))
      
    else:
      for filename in file_list:
        path = os.path.join(dataset_dir, filename)
        tensor = np.load(path)
        tensor = torch.tensor(tensor)
        label_name = filename[0:3]
        label = self.classes.index(label_name)
        self.data.append((tensor,48000))
        self.labels.append(torch.tensor(label))

    self.device = device
    
    '''
    if feature_extract:
      feature_extracto = feature_extractor(self)
      feature_extracto.apply()
      feature = feature_extracto.get_features()
      for i in range(len(self.data)):
        self.data[i] = (feature[i].unsqueeze(0), self.data[i][1])
    ''' 

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