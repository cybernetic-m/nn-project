import torch
import gc

def train_one_epoch (training_loader, model, loss_fn, optimizer):

    # Initialization of lists (predictions and true values) and the value of loss_epoch
    y_pred_list = []
    y_true_list = []
    loss_epoch = 0

    for i, data in enumerate (training_loader):
        
        # Divide the tuple data in inputs tensor and labels tensor (batch_size dimension)
        #print(data)
        x, y_true = data
        
        # extract waveform from tuple (waveform, sample_rate)
        #print("x_data",x)
        x = x[0]
        device = x[0].device
        
        # Put the gradient to zero for every batch
        optimizer.zero_grad()

        # Make predictions
        y_pred = model(x)

        # Compute the loss and the gradient
        #y_pred = torch.argmax(y_pred, dim)
        #print("y_pred", y_pred)
        #print("y_true", y_true)
        loss_value = loss_fn(y_pred, y_true)

        loss_value.backward()

        # Update the weights
        optimizer.step()

        #print("loss_report", loss_report)
        # Summing the values of the loss in each batch of the epoch
        loss_epoch += loss_value.detach().item()

        # Add the "batch" predictions and true values to the corrispettive lists
        #print("y_true", y_true.tolist())
        #y_true_tmp = torch.argmax(y_true).cpu().item()
        #print("y_true_tmp", y_true_tmp)
        y_true_list += y_true.cpu().tolist()

        #print("y_pred", y_pred.cpu().tolist())
        y_pred_list += torch.argmax(y_pred.detach(), dim=1).cpu().tolist()

        del x, y_true, y_pred, loss_value
        gc.collect()

        '''
        print("y_true_list", y_true_list)
        print("y_pred_list", y_pred_list)
        print("y_pred", y_pred)
        print("torch.argmax(y_pred)", torch.argmax(y_pred))
        '''

    

    if device == "cuda":
        torch.cuda.empty_cache()

    loss_avg = loss_epoch / len(training_loader)   
    
    return loss_avg, y_pred_list, y_true_list

 





