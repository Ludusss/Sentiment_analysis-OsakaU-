import time

import optuna

from model import MLP
import torch
import numpy as np
from utils import process_ESD_features, report_acc_mlp
from optuna.trial import TrialState


DEVICE = torch.device("cpu")
audio_train, audio_labels_train, audio_test, audio_labels_test, audio_val, audio_labels_val = process_ESD_features()

time.sleep(1000)

def objective(trial):
    # Generate the model.
    audio_model_info = torch.load("saved_models/audio_mlp/ESD/3_model_ESD_acc_84.85.a")
    model = MLP(input_feature_size=33, hidden_size=118, n_classes=3,
                      n_layers=1, device=DEVICE)
    model.load_state_dict(audio_model_info['model_state_dict'])

    weight_fc = trial.suggest_float("weight_fc", -0.001, 0.6)
    weight_fc1 = trial.suggest_float("weight_fc1", -0.001, 0.6)
    weight_fc2 = trial.suggest_float("weight_fc", -0.001, 0.6)
    weight_fc12 = trial.suggest_float("weight_fc1", -0.001, 0.6)

    for name, param in model.state_dict().items():
        if name == "fc.weight":
            param[:][0] = param[:][0] + weight_fc
        if name == "fc1.weight":
            param[:][0] = param[:][0] + weight_fc1

    model.eval()
    with torch.no_grad():
        output_val = model(torch.Tensor(np.array(audio_val)).to(DEVICE))
        acc_test, f1_test, conf_matrix, classify_report = report_acc_mlp(output_val, torch.Tensor(audio_labels_val).to(DEVICE))

    return acc_test

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=700, timeout=600)

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
