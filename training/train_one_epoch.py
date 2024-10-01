import torch

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
        print("x[0]",x.shape)

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

        loss_report = loss_value.detach()
        #print("loss_report", loss_report)
        # Summing the values of the loss in each batch of the epoch
        loss_epoch += loss_report

        # Add the "batch" predictions and true values to the corrispettive lists
        #print("y_true", y_true.tolist())
        #y_true_tmp = torch.argmax(y_true).cpu().item()
        #print("y_true_tmp", y_true_tmp)
        y_true_list += y_true.tolist()

        #print("y_pred", y_pred.cpu().tolist())
        y_pred_tmp = y_pred.detach().cpu().tolist()
        y_pred_li = []
        for y_pred_l in y_pred_tmp:
            #print("y_pred_l",type(y_pred_l))
            y_pred_li += [y_pred_l.index(max(y_pred_l))]
        #print("y_pred_li", y_pred_li)
        y_pred_list += y_pred_li
        '''
        print("y_true_list", y_true_list)
        print("y_pred_list", y_pred_list)
        print("y_pred", y_pred)
        print("torch.argmax(y_pred)", torch.argmax(y_pred))
        '''
    
    loss_avg = loss_epoch / i   # tensor(value, device = 'cuda:0')
    loss_avg = loss_avg.item() # Take only value
    
    return loss_avg, y_pred_list, y_true_list

 





