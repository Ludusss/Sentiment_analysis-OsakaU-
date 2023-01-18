import torch
import optuna
from optuna.trial import TrialState
import torch.nn as nn
import os

from torch.optim.lr_scheduler import StepLR

from model import MLP
from utils import process_ESD_features, gen_batches_mlp
import numpy as np

DEVICE = torch.device("cpu")
CLASSES = 3
DIR = os.getcwd()
EPOCHS = 30

def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 4)
    layers = []

    in_features = 33
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))
        in_features = out_features

    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.Softmax(dim=1))

    return nn.Sequential(*layers)


def objective(trial):
    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_decay = trial.suggest_float("lr_decay", 0.1, 0.99, log=True)
    scheduler = StepLR(optimizer, step_size=1, gamma=lr_decay)

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # Get the audio data
    audio_train, audio_labels_train, audio_test, audio_labels_test, audio_val, audio_labels_val = process_ESD_features()

    # Batch and epochs optimization
    batch_size = trial.suggest_int("batch_size", 1, audio_train.shape[0])



    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        batches = gen_batches_mlp(audio_train, audio_labels_train, batch_size)
        for idx, batch in enumerate(batches):
            b_train_audio, b_train_label = zip(*batch)
            input_audio = torch.Tensor(np.array(b_train_audio)).to(DEVICE)
            target_train = torch.Tensor(np.array(b_train_label)).to(DEVICE)
            target_train = target_train.view(-1).long()

            # Forward pass
            output_audio = model(input_audio)
            loss = criterion(output_audio, target_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))