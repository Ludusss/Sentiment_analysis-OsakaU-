import time
import sys
sys.path.append('/Users/ludus/Projects/Sentiment_analysis-OsakaU-/luo')
import optuna
from nltk.sentiment import SentimentIntensityAnalyzer

from luo.model import LstmModel, hir_fullModel
from model import MLP
import torch
import numpy as np
from utils import process_ESD_features, report_acc_mlp, process_features, report_acc
from optuna.trial import TrialState


#audio_train, audio_labels_train, audio_test, audio_labels_test, audio_val, audio_labels_val = process_ESD_features()
#_, _, _, _, _, _, _, _, _, audio_features_test, audio_labels_test, audio_mask_test, _, _ = process_features(True)
_, _, _, text_test, text_test_label, text_test_mask, _, _, _, audio_test, audio_test_label, audio_mask_test, _, _ = process_features(True)
input_data_esd = torch.Tensor(np.reshape(audio_test, (3286, 33)))
input_labels_esd = torch.Tensor(np.reshape(audio_test_label, 3286))
input_data_audio = torch.Tensor(audio_test)
input_data_text = torch.Tensor(text_test)

def objective(trial):
    # Generate the model.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = LstmModel(input_feat_size=33, output_size=4, hidden_dim=300, fc_dim=200, dropout=0.5)
    #model.load_state_dict(torch.load("./luo/state_dict/audioRNN/audioRNN50.19.pt", map_location=device))
    model = hir_fullModel(batch_size=10, mode=False, classifier="lstm", output_size=4, hidden_dim=300, fc_dim=200,
                          dropout=0.5, n_layers=2)
    model.load_state_dict(torch.load("./luo/state_dict/full/full73.14.pt", map_location=device));
    esd_model = MLP(input_feature_size=33, hidden_size=118, n_classes=4,
                    n_layers=1, device=device, p=0.2554070776341251)
    model_info = torch.load("./saved_models/audio_mlp/ESD/4/4_model_ESD_acc_80.06k=0.a", map_location=device)
    esd_model.load_state_dict(model_info["model_state_dict"])

    threshold_happy = trial.suggest_float("t_happy", 0.1, 1.0)
    threshold_sad = trial.suggest_float("t_sad", 0.1, 1.0)
    threshold_angry = trial.suggest_float("t_angry", 0.1, 1.0)
    threshold_neutral = trial.suggest_float("t_neutral", 0.1, 1.0)

    model.eval()
    esd_model.eval()
    with torch.no_grad():
        output_baseline, _ = model(input_data_audio, input_data_text)
        output_baseline = torch.softmax(output_baseline, dim=1)

        for i, label_probs in enumerate(output_baseline):
            label = label_probs.argmax(dim=0, keepdims=True).item()
            match label:
                case 0:
                    if label_probs[label] < threshold_angry:
                        output = esd_model(input_data_esd[i])
                        output = torch.softmax(output, dim=0)
                        output_baseline[i] = output
                case 1:
                    if label_probs[label] < threshold_happy:
                        output = esd_model(input_data_esd[i])
                        output = torch.softmax(output, dim=0)
                        output_baseline[i] = output
                case 2:
                    if label_probs[label] < threshold_sad:
                        output = esd_model(input_data_esd[i])
                        output = torch.softmax(output, dim=0)
                        output_baseline[i] = output
                case 3:
                    if label_probs[label] < threshold_neutral:
                        output = esd_model(input_data_esd[i])
                        output = torch.softmax(output, dim=0)
                        output_baseline[i] = output

        acc_test, f1_test, conf_matrix, classify_report = report_acc(output_baseline, torch.Tensor(text_test_label).to(device), audio_mask_test)

    return acc_test

if __name__ == "__main__":
    search_space = {"t_angry": np.linspace(0.1, 0.9, num=10), "t_happy": np.linspace(0.1, 0.9, num=10), "t_sad": np.linspace(0.1, 0.9, num=50), "t_neutral": np.linspace(0.1, 0.9, num=50)}
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=1000, timeout=600)

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
