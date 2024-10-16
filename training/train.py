import os
import sys
import torch

# Get the absolute paths of the directories containing the utils functions and train_one_epoch
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
training_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../training'))

# Add these directories to sys.path
sys.path.append(utils_path)
sys.path.append(training_path)

# Import section
from utils import calculate_metrics, save_metrics, save_hydra_config
from train_one_epoch import train_one_epoch

def train (num_epochs, training_metrics_dict, validation_metrics_dict, training_loader, validation_loader, model, loss_fn, optimizer, plateau_scheduler, cfg):
    
    best_vloss = 10000000000

    for epoch in range(num_epochs):
        
        print(f'EPOCH {epoch + 1}/{num_epochs}:')

        # Compute the average loss and the predictions vs true values
        # Train the model for the single epoch
        model.train()
        loss_avg, y_pred_list, y_true_list = train_one_epoch(training_loader, model, loss_fn, optimizer)

        # Calculate the metrics
        print("TRAIN:")
        calculate_metrics(y_true_list, y_pred_list, training_metrics_dict, loss_avg, epoch) 
        
        # Validation part (disable the gradient computation)
        # Set the model in validation mode
        model.eval() 
        with torch.no_grad():
            vy_pred_list = []
            vy_true_list = []
            vloss_epoch = 0
            for i, vdata in enumerate (validation_loader):
                vx, vy_true = vdata
                vx = vx[0]
                vy_pred = model(vx)
                #print("vy_pred",vy_pred)
                #print("vy_true",vy_true)                
                vloss = loss_fn(vy_pred, vy_true)
                #print("vloss",vloss)
                vloss_epoch += vloss.detach().item()
                vy_true_list += vy_true.cpu().tolist()
                vy_pred_list += torch.argmax(vy_pred.detach(), dim=1).cpu().tolist()
            #print("vloss_epoch", vloss_epoch)  
            vloss_avg = vloss_epoch / len(validation_loader)
            
            plateau_scheduler.step(vloss_avg)

            print("VALIDATION:")
            calculate_metrics(vy_true_list, vy_pred_list, validation_metrics_dict, vloss_avg, epoch)

        print(f'LOSS train {loss_avg} valid {vloss_avg}')

        if vloss_avg < best_vloss:
            best_vloss = vloss_avg
            save_path = os.path.join(training_path, "results")
            parent_dir = model.save(save_path) # the name_path is: "your_path/2024-06-25_14:06:10/model.pt"
    
    results_path = parent_dir

    name_training_metrics = '/training_metrics.json'
    name_validation_metrics = '/validation_metrics.json'

    save_metrics(training_metrics_dict, results_path + name_training_metrics)
    save_metrics(validation_metrics_dict, results_path + name_validation_metrics)
            
    save_hydra_config(cfg, results_path+'/parameters.txt')