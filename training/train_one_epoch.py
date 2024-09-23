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
        print("x_data",x)
        x = (x[0].T, x[1])
        print("x[0].T_data",x)

        # Put the gradient to zero for every batch
        optimizer.zero_grad()

        # Make predictions
        y_pred = model(x)

        # Compute the loss and the gradient
        loss_value = loss_fn(y_pred, y_true)
        loss_fn.backward()

        # Update the weights
        optimizer.step()

        # Summing the values of the loss in each batch of the epoch
        loss_epoch += loss_value

        # Add the "batch" predictions and true values to the corrispettive lists
        y_true_list += y_true.tolist()
        y_pred_list += y_pred.tolist()
    
    # Average Loss in the epoch
    loss_avg = loss_epoch / i 
    
    return loss_avg, y_pred_list, y_true_list

 





