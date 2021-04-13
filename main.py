import torch
from Network import Transformer
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from TF_training import transformer_train, test_transformer
from functools import partial
import os
import sys
import json
from csv import DictReader

def load_and_test():
    main_path = "/data/results/vcpujol/transformers/single_deployment/predict_maxmax/pytorch_transformer/"

    summary_file = "to_check_results.csv"
    file = main_path + summary_file

    data = DictReader(open(file), delimiter=",", fieldnames=["test", "name", "chkpt", "loss"])

    next(data)
    for info in data:
        data_path = main_path + info["test"] + "/" + info["name"]
        checkpoint_id = info["chkpt"] + "/"
        loss_f = info["test"].split("_")[0]
        with open(data_path + "/params.json") as jfile:
            config = json.load(jfile)

        best_trained_model = Transformer(config["dim_val"], config["dim_att"],
                                         config["input_feat_enc"], config["input_feat_dec"],
                                         config["seq_len"], config["decoder_layers"],
                                         config["encoder_layers"], config["n_heads"])

        checkpoint = data_path + "/" + checkpoint_id + "checkpoint"

        # best_trained_model.to(device)
        model_state, _ = torch.load(checkpoint, map_location='cpu')
        best_trained_model.load_state_dict(model_state)
        pytorch_total_params = sum(p.numel() for p in best_trained_model.parameters() if p.requires_grad)
        print(pytorch_total_params)

        save_dir = "/data/results/vcpujol/transformers/single_deployment/predict_maxmax/pytorch_transformer/extra_figures/"
        experiment_name = info["test"] + "_" + info["name"]

        test_transformer(best_trained_model, config, save_dir, experiment_name, loss_f)


def main(loss_function="L1", num_samples=25, max_num_epochs=25, gpus_per_trial=1, cpus_per_trial=10):

    experiment_name = loss_function + "_shuffle_validation"
    save_dir = '/data/results/vcpujol/transformers/single_deployment/predict_maxmax/pytorch_transformer/'

    config = {
        "lr": tune.loguniform(1e-4, 5e-1),
        "lr_step": tune.randint(1,10),
        "gamma": tune.loguniform(0.85,0.9999),
        "epochs": tune.choice([5, 10, 15, 20, 25]),
        "n_heads": tune.randint(2,10),
        "dim_val": tune.choice([2,4,6]), # FIXME requires numero parell...
        "dim_att": tune.randint(2,12),
        "encoder_layers": tune.randint(1,7),
        "decoder_layers": tune.randint(1,7),
        "batch_size": tune.randint(1,10),
        "input_feat_enc": tune.choice([94]),
        "input_feat_dec": tune.choice([1]),
        "seq_len": tune.choice([16, 32, 64, 96, 128, 180, 220, 256, 312, 350, 420, 470, 512]),  #[16, 32, 64, 128, 256, 512, 1024, 2048]
        "prediction_step": tune.choice([1])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=4,
        reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=["lr", "lr_step", "gamma", "epochs", "n_heads", "dim_val", "dim_att", "encoder_layers",
                           "decoder_layers", "batch_size", "seq_len"],
        metric_columns=["loss", "training_iteration"])
    result = tune.run(
        partial(transformer_train, save_dir=save_dir, loss_function=loss_function),
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=save_dir,
        name=experiment_name)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    # print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

    best_trained_model = Transformer(best_trial.config["dim_val"], best_trial.config["dim_att"],
                                     best_trial.config["input_feat_enc"], best_trial.config["input_feat_dec"],
                                     best_trial.config["seq_len"], best_trial.config["decoder_layers"],
                                     best_trial.config["encoder_layers"], best_trial.config["n_heads"])

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    local_dir = save_dir
    exp_name = experiment_name
    test_acc = test_transformer(best_trained_model, best_trial.config, local_dir, exp_name, loss_function)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == '__main__':
    print(torch.__version__)

    loss_function = None
    if len(sys.argv) != 2:
        print("too much or no loss function specified.")
        print("Options are: L1, L2, SL")
        exit(1)
    else:
        if "L1" == sys.argv[1]:
            loss_function = "L1"
        elif "L2" == sys.argv[1]:
            loss_function = "L2"
        elif "SL" == sys.argv[1]:
            loss_function = "SL"
        else:
            print("loss function not implemented. Defaults to L1")
            loss_function = "L1"

    main(loss_function)

    print("done")
