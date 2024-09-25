import torch

def testing(model, model_path, test_loader, test_metrics):

    # Set the model in evaluation mode
    model.eval_mode()
    
    # Load the weights
    model.load(model_path)

    

    for data in test_loader:
        x, y_true = test_loader 


