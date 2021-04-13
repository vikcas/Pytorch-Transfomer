from Network import Transformer
import torch
from data_loader import LoadAggregatedAzureDataset
from torch.utils.data import DataLoader
from ray import tune
import os
import matplotlib.pyplot as plt
from torchsummary import summary


def transformer_train(config, save_dir, loss_function):

    # Learning hyperparams
    lr = config['lr']
    lr_step = config['lr_step']
    gamma = config['gamma']
    epochs = config['epochs']

    # Model hyperparams
    n_heads = config['n_heads']
    dim_val = config['dim_val']
    dim_attn = config['dim_att']
    n_decoder_layers = config['decoder_layers']
    n_encoder_layers = config['encoder_layers']

    # Data hyperparams
    batch_size = config['batch_size']
    input_feat_enc = config['input_feat_enc']
    input_feat_dec = config['input_feat_dec']
    seq_len = config['seq_len']
    prediction_step = config['prediction_step']

    #init network and optimizer
    t_model = Transformer(dim_val, dim_attn, input_feat_enc, input_feat_dec, seq_len, n_decoder_layers, n_encoder_layers, n_heads)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        #if torch.cuda.device_count() > 1:
        #    t_model = torch.nn.DataParallel(t_model)
    t_model.to(device)

    optimizer = torch.optim.Adam(t_model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=lr_step, gamma=gamma)
    if loss_function == "L1":
        loss_method = torch.nn.L1Loss()
    elif loss_function == "L2":
        loss_method = torch.nn.MSELoss()
    elif loss_function == "SL":
        loss_method = torch.nn.SmoothL1Loss()
    else:
        loss_method = torch.nn.L1Loss()
    # l1_loss = torch.nn.L1Loss()
    # l2_loss = torch.nn.MSELoss()
    #kl_criterion = torch.nn.KLDivLoss(reduction='batchmean')  # Parameters to check

    # Datasets
    azure_train_dataset = LoadAggregatedAzureDataset('train', seq_len, prediction_step)
    azure_dataloader_train = DataLoader(azure_train_dataset, batch_size, shuffle=False)
    azure_val_dataset = LoadAggregatedAzureDataset('validation', seq_len, prediction_step)
    azure_dataloader_val = DataLoader(azure_val_dataset, batch_size=1, shuffle=False)

    for e in range(epochs):
        out = []

        # Train model
        running_loss = 0.0
        epoch_steps = 0
        ii = 0
        for x_enc, x_dec, y in azure_dataloader_train:
            optimizer.zero_grad()
            x_dec = x_dec.unsqueeze(-1)
            x_enc, x_dec, y = x_enc.to(device), x_dec.to(device), y.to(device)
            out = t_model.forward(x_enc.float(), x_dec.float())

            # Recover scale to compute loss... TODO: is this good?
            # min_max_scaler = load("/data/cloud_data/AzurePublicDataset2019/processed_data/univariate_data/depl_ANu_all/min_max_scaler.joblib")
            # y_expanded = np.zeros([2,95])
            # out_expanded = np.zeros([2,95])
            # y_expanded[:,-1] = y.squeeze().cpu()
            # out_expanded[:,-1] = out.squeeze().cpu()
            #
            # y_n = min_max_scaler.inverse_transform(y_expanded)[:, -1]
            # out_n = min_max_scaler.inverse_transform(out_expanded)[:, -1]

            #loss = l1_loss(out, y)
            loss = loss_method(y.double(), out.double())
            loss.backward()
            optimizer.step()

            # running_loss += loss.item()
            # epoch_steps += 1
            # if ii % 500 == 0:  # print every 100 mini-batches
            #     print("[%d, %5d] loss: %.8f" % (e + 1, ii + 1, running_loss / epoch_steps))
            #     running_loss = 0.0 # TODO: Check if commented or not...
            # ii = ii + 1

        # Validate model
        val_loss = 0.0
        val_steps = 0
        for x_enc, x_dec, y in azure_dataloader_val:
            with torch.no_grad():
                x_dec = x_dec.unsqueeze(-1)
                x_enc, x_dec, y = x_enc.to(device), x_dec.to(device), y.to(device)
                out = t_model.forward(x_enc.float(), x_dec.float())

                #loss = l1_loss(out,y)
                #loss = kl_criterion(out.double(), y.double())
                loss = loss_method(y.double(), out.double())

                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(e) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((t_model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps))
        lr_scheduler.step()
    print("Finished Training")

def test_transformer(t_model, config, save_dir, experiment, loss_function):

    device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:1"
    #     #if torch.cuda.device_count() > 1:
    #     #    t_model = torch.nn.DataParallel(t_model)
    # t_model.to(device)

    if loss_function == "L1":
        loss_method = torch.nn.L1Loss()
    elif loss_function == "L2":
        loss_method = torch.nn.MSELoss()
    elif loss_function == "SL":
        loss_method = torch.nn.SmoothL1Loss()
    else:
        loss_method = torch.nn.L1Loss()
    #l1_criterion = torch.nn.L1Loss()
    #kl_criterion = torch.nn.KLDivLoss(reduction='batchmean') # Parameters to check
    # l2_loss = torch.nn.MSELoss()

    test_dataset = LoadAggregatedAzureDataset('test', config["seq_len"], config["prediction_step"])

    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    objective_data = list()
    obtained_data = list()
    loss_progression = list()

    ii = 0
    for x_enc, x_dec, y in testloader:
        with torch.no_grad():
            x_dec = x_dec.unsqueeze(-1)
            x_enc, x_dec, y = x_enc.to(device), x_dec.to(device), y.to(device)

            # if ii == 0:
            #     summary(t_model, [(94,1,1),(1,1,1)])

            out = t_model.forward(x_enc.float(), x_dec.float())


            #loss = l1_criterion(out, y)
            #loss = kl_criterion(out.double(), y.double())
            loss = loss_method(y.double(), out.double())

            objective_data.append(y.squeeze().tolist())
            obtained_data.append(out.squeeze().tolist())
            loss_progression.append(loss.cpu().detach().tolist())
            ii = ii + 1

    mean_loss = sum(loss_progression)/len(loss_progression)
    plt.figure(figsize=(30, 10))
    plt.plot(objective_data, '-', color='indigo', label='Target', linewidth=2)
    plt.plot(obtained_data, '--', color='limegreen', label='Forecast', linewidth=2)
    plt.legend()
    plt.title("Prediction in test. Avg loss: " + f'{mean_loss:.4f}')
    plt.savefig(save_dir + experiment + "_Prediction.png")
    #plt.savefig(save_dir + experiment + "/Prediction.png")
    plt.close()

    plt.figure(figsize=(20, 8))
    if len(loss_progression) < 2:
        plt.scatter(1, loss_progression, label='Loss')
    else:
        plt.plot(loss_progression, '-', color='indigo', label='Loss', linewidth=2)
    plt.legend()
    plt.title("Loss evolution in test")
    plt.savefig(save_dir + experiment + "_loss.png")
    # plt.savefig(save_dir + experiment + "/loss.png")
    plt.close()

    return mean_loss