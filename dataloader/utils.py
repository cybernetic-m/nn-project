import os
from tqdm import tqdm
import zipfile
import shutil
import random
import json
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torchaudio
import torch
from dataloader.preprocessing import pitch_and_speed_perturbation, speed_perturbation, SpecAugmentFreq, SpecAugmentTime
import gc

def download_dataset (link_dataset, destination_dir, gdrive_link, extract_dir):
  file_id = os.path.split(link_dataset)[0].split('/')[-1]  # Take the file_id (Ex. "https://drive.google.com/file/d/1BMj4BGXxIMzsd-GYSAEMpB7CF0XB87UT/view?usp=sharing" => file_id: 1BMj4BGXxIMzsd-GYSAEMpB7CF0XB87UT)
  download_link = gdrive_link + file_id # Create the path for gdown (Ex. https://drive.google.com/uc?id={YOUR_FILE_ID})

  # Try to download the dataset if the dataset was not already downloaded
  try:
    import gdown # type: ignore
    if not(os.path.exists(extract_dir)):
      gdown.download(
          download_link,
          destination_dir,
          quiet=False)
    else:
      print("Dataset already downloaded")

  except Exception as error:
    print ("Error in downloading:")
    print(error)

  # Try to unzip the zipped file downloaded
  try:
    if not(os.path.exists(extract_dir)):
      with zipfile.ZipFile(destination_dir, 'r') as zip:

        os.makedirs(extract_dir)

        list_file = zip.namelist()  # Create a list of all the file in the zipped (Ex. ['file1.txt', 'dir1/file2.txt', ... ])
        file_len = len(list_file) # Get the length of the list (Ex. 5)

        with tqdm(total = file_len, desc= "Unzipping...") as pbar:

          for file in list_file:

            zip.extract(file, extract_dir)
            pbar.update(1)

          os.remove(destination_dir)
    else:
      print("Dataset already unzipped")

  except Exception as error:
    print ("Error in unzipping:")
    print(error)
    


def counter_classes (classes, dataset_dir):

  n_file = {class_:0 for class_ in classes}

  for class_ in classes:
    class_dir = os.path.join(dataset_dir, class_)
    n_file[class_] = len(os.listdir(class_dir))

  return n_file
  
  
def dataset_split(dataset_dir, extract_dir, train_perc, test_perc, val_perc):

  split_dir = ['train', 'test', 'val']
  new_dataset_dir = os.path.join(extract_dir, 'EMOVO_split')
  os.makedirs(new_dataset_dir, exist_ok=True)
  for split in split_dir:
    os.makedirs(os.path.join(new_dataset_dir, split), exist_ok=True)

  classes = ['dis', 'gio', 'neu', 'pau', 'rab', 'sor', 'tri']
  tmp_dir = os.path.join(extract_dir, 'EMOVO_tmp')
  os.makedirs(tmp_dir, exist_ok=True)
  for class_ in classes:
    os.makedirs(os.path.join(tmp_dir, class_), exist_ok=True)

  # Merge the entire Dataset
  try:
    if (os.path.exists(dataset_dir)):

      subdirectories = os.listdir(dataset_dir)

      for subdir in subdirectories:
        if not(subdir == 'documents'):
          for filename in os.listdir(os.path.join(dataset_dir, subdir)):
            filename_class = filename.split('-')[0]
            file_path = os.path.join(dataset_dir, subdir, filename)
            new_path = os.path.join(tmp_dir, filename_class, filename)

            # Move the file to the merged directory
            shutil.move(file_path, new_path)

      n_file = counter_classes(classes, tmp_dir)

      # Data Split randomly
      for class_ in classes:
        random_vector = random.sample(range(1, n_file[class_] + 1), n_file[class_])

        n_train = (n_file[class_] * train_perc)
        n_test = (n_file[class_] * test_perc)
        n_val = (n_file[class_] * val_perc)

        train_index = random_vector[:int(n_train)]
        test_index = random_vector[int(n_train):int(n_train) + int(n_test)]
        val_index = random_vector[int(n_train) + int(n_test):]

        for index in train_index:
          filename = os.listdir(os.path.join(tmp_dir, class_))
          filename_i = filename[index-1]
          source_dir = os.path.join(tmp_dir, class_, filename_i)
          target_dir = os.path.join(new_dataset_dir, 'train', filename_i)
          shutil.copy(source_dir, target_dir)

        for index in test_index:
          filename = os.listdir(os.path.join(tmp_dir, class_))
          filename_i = filename[index-1]
          source_dir = os.path.join(tmp_dir, class_, filename_i)
          target_dir = os.path.join(new_dataset_dir, 'test', filename_i)
          shutil.copy(source_dir, target_dir)

        for index in val_index:
          filename = os.listdir(os.path.join(tmp_dir, class_))
          filename_i = filename[index-1]
          source_dir = os.path.join(tmp_dir, class_, filename_i)
          target_dir = os.path.join(new_dataset_dir, 'val', filename_i)
          shutil.copy(source_dir, target_dir)

    else:
      print("Dataset not found")

  except Exception as error:
    print ("Error in dataset reordering:")
    print(error)

def augment_data(preprocess_pipeline, spectogram_pipeline, dataset_dir):
  save_dir = dataset_dir+'_aug'
  dataset_dir +='_split'
  try:
    if (os.path.exists(dataset_dir)) and not(os.path.exists(save_dir)) :
      
      subdir = 'train'
      if not(os.path.exists(save_dir)):
        os.makedirs(os.path.join(save_dir, subdir))

      for filename in os.listdir(os.path.join(dataset_dir, subdir)):
        file_path = os.path.join(dataset_dir, subdir, filename)
        save_path = os.path.join(save_dir, subdir, filename)
        
          
        waveform, sample_rate = torchaudio.load(file_path)
        with torch.no_grad():
          waveformPSP = pitch_and_speed_perturbation(waveform, sample_rate, 1.2, 2)
          save_path = save_path[:-4]+'-PSP.wav'
          torchaudio.save(save_path, waveformPSP, sample_rate)
          del waveformPSP
          gc.collect()
        with torch.no_grad():
          waveformSP = speed_perturbation(waveform, sample_rate, 1.2)
          save_path = save_path[:-4]+'-SP.wav'
          torchaudio.save(save_path, waveformSP, sample_rate)
          del waveformSP
          gc.collect()
        with torch.no_grad():
          waveformSAP = SpecAugmentFreq(waveform, sample_rate, 40)
          save_path = save_path[:-4]+'-SAP.wav'
          torchaudio.save(save_path, waveformSAP, sample_rate)
          del waveformSAP
          gc.collect()
        with torch.no_grad():
          waveformSAT = SpecAugmentTime(waveform, sample_rate, 40)
          save_path = save_path[:-4]+'-SAT.wav'
          torchaudio.save(save_path, waveformSAT, sample_rate)
          del waveform, waveformSAT, sample_rate
          gc.collect()
        #spectogramPSP = spectogram_pipeline(waveformPSP, sample_rate)
        #spectogramSP = spectogram_pipeline(waveformSP, sample_rate)
        #spectogramSP = spectogram_pipeline(waveformSAP, sample_rate)
        #spectogramSAT = spectogram_pipeline(waveformSAT, sample_rate)





        #del spectogramPSP, spectogramSP, spectogramSP, spectogramSAT

    else:
      print("Dataset not found or already augmented")

  except Exception as error:
    print(error)

#  return spectogramPSP, spectogramSP, spectogramSAP, spectogramSAT


def save_metrics(metrics, path):
  with open(path, "w") as file:
    json.dump(metrics, file, indent=4)
    print("Metrics saved:", path)

def load_metrics(path):
  if os.path.exists(path):
    with open(path, 'r') as file:
      data = json.load(file)
      print("Metrics loaded", path)
      return data
  else:
    print("The file", path, "does not exists!")
    
def calculate_metrics(y_true, y_pred, metrics_dict, loss, epoch, test=False):

  current_acc = metrics.accuracy_score(y_true, y_pred)
  current_prec = metrics.precision_score(y_true, y_pred, average='micro')
  current_recall = metrics.recall_score(y_true, y_pred, average='micro')
  current_f1_score = metrics.f1_score(y_true, y_pred, average='micro')
  if test == False:
    metrics_dict["epoch"].append(epoch)
  metrics_dict["loss"].append(loss)
  metrics_dict["accuracy"].append(current_acc)
  metrics_dict["precision"].append(current_prec)
  metrics_dict["recall"].append(current_recall)
  metrics_dict["f1_score"].append(current_f1_score)

  print(f"accuracy: {current_acc*100:.2f}, precision: {current_prec*100:.2f}, recall: {current_recall*100:.2f}, f1-score: {current_f1_score*100:.2f}")


def plot_confusion_matrix(cm, class_names, normalize=False, title='Confusion Matrix', cmap='Oranges'):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names)

    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()


def plot_loss_acc (epochs, training_loss, validation_loss, training_accuracy, validation_accuracy):

  # Is it necessary to use numpy?
  '''
  epochs = np.linspace(0, len(epochs), 1)
  
  training_loss = np.array(training_loss)
  validation_loss = np.array(validation_loss)
  training_accuracy = np.array(training_accuracy)
  validation_accuracy = np.array(validation_accuracy)
  '''

  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,6))

  # Set the functions, title, x_label, y_label and legend for the loss 
  ax[0].plot(epochs, training_loss, label='Training Loss', color='b')
  ax[0].plot(epochs, validation_loss, label='Validation Loss', color='r')
  ax[0].set_title('Loss')
  ax[0].set_xlabel('Epochs')
  ax[0].set_ylabel('Loss value')
  ax[0].legend()

  # Set the functions, title, x_label, y_label and legend for the accuracy
  ax[1].plot(epochs, training_accuracy, label='Training Accuracy', color='b')
  ax[1].plot(epochs, validation_accuracy, label='Validation Accuracy', color='r')
  ax[1].set_title('Accuracy')
  ax[1].set_xlabel('Epochs')
  ax[1].set_ylabel('Accuracy value')
  ax[0].legend()

# Display the plot
plt.tight_layout()  # This helps to prevent overlapping of subplots
plt.show()
