import os 
import sys


path = './dataset/EMOVO_aug'
    
path_train = os.path.join(path,'train')
path_test = os.path.join(path,'test')
path_val = os.path.join(path,'val')

list_file_train = os.listdir(path_train)
list_file_test = os.listdir(path_test)
list_file_val = os.listdir(path_val)

check_test_train = set(list_file_test) == set(list_file_train)
check_test_val = set(list_file_test) == set(list_file_val)
check_train_val = set(list_file_train) == set(list_file_val)
print(check_test_train)
print(check_test_val)
print(check_train_val)


if not (check_test_train and check_test_val and check_train_val):
    print("OK!")





    
    
        
    
