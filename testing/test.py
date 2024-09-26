import os
import sys
import torch
import sklearn.metrics

# Get the absolute paths of the directories containing the utils functions and testing directory
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
testing_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../testing'))

# Add these directories to sys.path
sys.path.append(utils_path)

# Import section
from utils import calculate_metrics, save_metrics

def test(model, model_path, test_dataloader, test_metrics_dict, loss_fn):

    # Set the model in evaluation mode
    model.eval_mode()
    
    # Load the weights
    model.load(model_path)

    loss = 0  # value of the test loss
    # Lists of predictions and true labels
    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):

            # Take input and label from the list [input, label]
            x, y_true = data       # data = [ tensor([[[waveform]]], [sample_rate]),   tensor([3], device='cuda:0')]
            
            # Take only the waveform for the input tuple (waveform, sample_rate)
            x = x[0]       # x[0] = tensor([[[waveform]]], [sample_rate])
            
            # Make predictions (Tensor with 7 values of probability, 1 for each class)
            y_pred = model(x)  # y_pred = tensor([[0.1254, 0.1152, 0.1396, 0.1600, 0.1058, 0.1394, 0.2146]],device='mps:0')
            
            # Compute the loss
            loss_value = loss_fn(y_pred, y_true)
            loss += loss_value # Incremental value for the average

            # Create the list of the y_true and y_pred
            # Transform the tensor([3], device='cuda:0') in a list [6] and summing the lists y_true_list = [6, 1, 2, 4 ...]
            y_true_list += y_true.tolist()  
            # Argmax take the index (position) of the maximum value in y_pred => torch.argmax(y_pred) = tensor(6, device='mps:0') and item() take only the value 6
            y_pred_tmp = torch.argmax(y_pred).item() 
            y_pred_list += [y_pred_tmp] # y_pred_list = [6, 4, 3, 2, ....]

    # Average Loss in testing
    loss_avg = loss / i   # tensor(value, device = 'cuda:0')
    loss_avg = loss_avg.item() # Take only value

    calculate_metrics(y_true_list, y_pred_list, test_metrics_dict, loss_avg, i, test=True)
   
    cm = sklearn.metrics.confusion_matrix(y_true=y_true_list, y_pred=y_pred_list, labels=[0,1,2,3,4,5,6])
    
    results_path = testing_path

    name_test_metrics = f'/{model.model_name}_test_metrics.json'
   
    save_metrics(test_metrics_dict, results_path + name_test_metrics)

    return cm







