import os
from tqdm import tqdm
import gdown
import zipfile
import shutil
import random

def download_dataset (link_dataset, destination_dir, gdrive_link, extract_dir):
  file_id = os.path.split(link_dataset)[0].split('/')[-1]  # Take the file_id (Ex. "https://drive.google.com/file/d/1BMj4BGXxIMzsd-GYSAEMpB7CF0XB87UT/view?usp=sharing" => file_id: 1BMj4BGXxIMzsd-GYSAEMpB7CF0XB87UT)
  download_link = gdrive_link + file_id # Create the path for gdown (Ex. https://drive.google.com/uc?id={YOUR_FILE_ID})

  # Try to download the dataset if the dataset was not already downloaded
  try:
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