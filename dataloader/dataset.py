from torch.utils.data import Dataset
import os
import torchaudio
import torch
from feature_extractor_tim import feature_extractor

class EMOVO_Dataset(Dataset):

  def __init__(self, dataset_dir, feature_extract=False, device = 'cpu'):

    self.dataset_dir = dataset_dir
    self.classes = ['dis', 'gio', 'neu', 'pau', 'rab', 'sor', 'tri']
    self.data = [] # Waw Audio as tensors
    self.labels = [] # Labels of the data
    
    extractor = feature_extractor() 

    file_list = os.listdir(dataset_dir)
    for filename in file_list:
      path = os.path.join(dataset_dir, filename)
      tensor, sample_rate = torchaudio.load(path)

      if tensor.shape[0] == 1: # if an audio is mono we duplicate the channel to make it stereo 
        tensor = torch.cat((tensor,tensor))
        print("fixed a mono track:",filename)
        torchaudio.save(path, tensor, sample_rate)

      if feature_extract:
        features = extractor(tensor)
        class_ = self.classes.index(filename.split('-')[0])
        self.data.append((features,sample_rate))
        print(features)
        self.labels.append(torch.tensor(class_))

      else:
        class_ = self.classes.index(filename.split('-')[0])
        self.data.append((tensor,sample_rate))
        self.labels.append(torch.tensor(class_))

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