import os
import sys
import torch

# Get the absolute paths of the directories containing the utils functions and train_one_epoch
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))

# Add these directories to sys.path
sys.path.append(utils_path)

# Import section
from utils import calculate_metrics, save_metrics

def testing(model, model_path, test_loader, test_metrics, loss_fn):

    # Set the model in evaluation mode
    model.eval_mode()
    
    # Load the weights
    model.load(model_path)

    loss = 0  # value of the test loss
    # Lists of predictions and true labels
    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):

            # Take input and label from the tuple (input, label)
            x, y_true = data 
            print ("x:", x)

            # Take only the waveform for the input tuple (waveform, sample_rate)
            x = x[0]
            print ("x:", x)

            # Make predictions
            y_pred = model(x)
            print ("y_pred:", y_pred)

            # Compute the loss
            loss_value = loss_fn(y_pred, y_true)
            loss += loss_value # Incremental value for the average

            y_true_list += y_true.tolist() 
            print ("y_true_list:", y_true_list)
            y_pred_tmp = torch.argmax(y_pred)
            print ("y_pred_tmp:", y_pred_tmp)
            y_pred_list += [y_pred_tmp]
            print ("y_pred_list:", y_pred_list)





    # Average Loss in testing
    loss_avg = loss / i 







