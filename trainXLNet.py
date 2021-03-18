from XLNet import *
from tqdm import trange
def trainXLNet(model, num_epochs, \
          optimizer, \
          train_dataloader, valid_dataloader, \
          model_save_path, \
          train_loss_set=[], valid_loss_set=[], \
          lowest_eval_loss=None, start_epoch=0, \
          device="cpu"
          ):
    """
    Train the model and save the model with the lowest validation loss
    """

    model.to(device)

    # trange is a tqdm wrapper around the normal python range
    for i in trange(num_epochs, desc="Epoch"):
        # if continue training from saved model
        actual_epoch = start_epoch + i

        # Training

        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0
        num_train_samples = 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            # store train loss
            tr_loss += loss.item()
            num_train_samples += b_labels.size(0)
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # scheduler.step()

        # Update tracking variables
        epoch_train_loss = tr_loss / num_train_samples
        train_loss_set.append(epoch_train_loss)

        print("Train loss: {}".format(epoch_train_loss))

        # Validation

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables
        eval_loss = 0
        num_eval_samples = 0

        # Evaluate data for one epoch
        for batch in valid_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate validation loss
                loss = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                # store valid loss
                eval_loss += loss.item()
                num_eval_samples += b_labels.size(0)

        epoch_eval_loss = eval_loss / num_eval_samples
        valid_loss_set.append(epoch_eval_loss)

        print("Valid loss: {}".format(epoch_eval_loss))

        if lowest_eval_loss == None:
            lowest_eval_loss = epoch_eval_loss
            # save model
            save_model(model, model_save_path, actual_epoch, \
                       lowest_eval_loss, train_loss_set, valid_loss_set)
        else:
            if epoch_eval_loss < lowest_eval_loss:
                lowest_eval_loss = epoch_eval_loss
                # save model
                save_model(model, model_save_path, actual_epoch, \
                           lowest_eval_loss, train_loss_set, valid_loss_set)
        print("\n")

    return model, train_loss_set, valid_loss_set


def save_model(model, save_path, epochs, lowest_eval_loss, train_loss_hist, valid_loss_hist):
    """
    Save the model to the path directory provided
    """
    model_to_save = model.module if hasattr(model, 'module') else model
    checkpoint = {'epochs': epochs, \
                  'lowest_eval_loss': lowest_eval_loss, \
                  'state_dict': model_to_save.state_dict(), \
                  'train_loss_hist': train_loss_hist, \
                  'valid_loss_hist': valid_loss_hist
                  }
    torch.save(checkpoint, save_path)
    print("Saving model at epoch {} with validation loss of {}".format(epochs, \
                                                                       lowest_eval_loss))
    return


def load_model(save_path):
    """
    Load the model from the path directory provided
    """
    checkpoint = torch.load(save_path)
    model_state_dict = checkpoint['state_dict']
    model = XLNetForMultiLabelSequenceClassification(num_labels=model_state_dict["classifier.weight"].size()[0])
    model.load_state_dict(model_state_dict)

    epochs = checkpoint["epochs"]
    lowest_eval_loss = checkpoint["lowest_eval_loss"]
    train_loss_hist = checkpoint["train_loss_hist"]
    valid_loss_hist = checkpoint["valid_loss_hist"]

    return model, epochs, lowest_eval_loss, train_loss_hist, valid_loss_hist