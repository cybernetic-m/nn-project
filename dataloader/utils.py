import os
import sys
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
import psutil
dataloader_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataloader'))
# Add these directories to sys.path
sys.path.append(dataloader_path)

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

def augment_data(dataset_src, extract_dir, transforms, device):
  process = psutil.Process(os.getpid())
  try:
    type_of_prep_list = ['white_noise', 'shifted', 'echo', 'pitch_speed'] 
    new_dataset_dir = os.path.join(extract_dir, 'EMOVO_aug')
    # Copy the entire directory to the new destination
    if not (os.path.exists(new_dataset_dir)):
      print("Copying...")
      shutil.copytree(dataset_src, new_dataset_dir)

      new_dataset_dir_train = os.path.join(new_dataset_dir, 'train')
      list_file = os.listdir(new_dataset_dir_train)

      for filename in list_file:
        memory_usage = process.memory_info().rss / (1024 ** 2)  # rss gives memory in bytes
        if memory_usage > 1024*20:
          os.kill(os.getpid(), 9)
        waveform, sample_rate = torchaudio.load(new_dataset_dir_train + '/' + filename)
        transformed_wave, type_of_prep = transforms(waveform.to(device), sample_rate)
        name_augmentation = type_of_prep_list[type_of_prep]
        new_name = new_dataset_dir_train + '/' + filename.split('.')[0] + '-' + name_augmentation + '.wav'
        torchaudio.save(new_name, transformed_wave.cpu(), sample_rate)

    else:
      print("The augmented dataset already exist!")

  except Exception as error:
    print ("Error in augmentation:")
    print(error)
  

# Function to save a tensor as a spectrogram image
def save_spectrogram_image(spectrogram, filename):
  # Convert the spectrogram tensor to numpy for plotting
  spectrogram = spectrogram.cpu().numpy()
  
  # Plot the spectrogram using matplotlib
  plt.figure(figsize=(10, 4))
  plt.imshow(spectrogram[0], origin='lower', aspect='auto', cmap='viridis')
  plt.colorbar(format='%+2.0f dB')
  
  # Save the image
  plt.savefig(filename)
  plt.close()

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

def smooth_label(target: int, num_classes: int, smoothing: float = 0.1):
    """
    Converts a single label to a smoothed label distribution.
    
    Args:
        target (int): The correct class label (index).
        num_classes (int): Total number of classes.
        smoothing (float): Smoothing factor (ε). Default is 0.1.
    
    Returns:
        torch.Tensor: A tensor of size (num_classes,) with the smoothed label distribution.
    """
    assert 0 <= target < num_classes, "Target label must be between 0 and num_classes - 1"
    
    if smoothing > 0.0:
        # Label smoothing logic
        smoothed_labels = torch.full((num_classes,), smoothing / (num_classes - 1))
        smoothed_labels[target] = 1.0 - smoothing
    else:
        # One-hot encoding logic (no smoothing)
        smoothed_labels = torch.zeros(num_classes)
        smoothed_labels[target] = 1.0
    
    return smoothed_labels
