import time
import sys
sys.path.append('/mman')
from mman.model import LstmModel, hir_fullModel
from model import MLP
import torch
import numpy as np
from utils import process_ESD_features, report_acc_mlp, process_features, report_acc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#_, _, _, _, _, _, _, _, _, audio_features_test, audio_labels_test, audio_mask_test, _, _ = process_features(True)
#input_data_esd = torch.Tensor(np.reshape(audio_features_test, (3286, 33)))
#input_labels_esd = torch.Tensor(np.reshape(audio_labels_test, 3286))
_, _, _, text_test, text_test_label, text_test_mask, _, _, _, audio_test, audio_test_label, audio_mask_test, _, _ = process_features(True)
input_data_esd = torch.Tensor(np.reshape(audio_test, (3286, 33)))
input_labels_esd = torch.Tensor(np.reshape(audio_test_label, 3286))
input_data_audio = torch.Tensor(audio_test)
input_data_text = torch.Tensor(text_test)

#model = LstmModel(input_feat_size=33, output_size=4, hidden_dim=300, fc_dim=200, dropout=0.5)
#model.load_state_dict(torch.load("./mman/state_dict/audioRNN/audioRNN50.19.pt", map_location=device))
model = hir_fullModel(batch_size = 10, mode = False, classifier = "lstm", output_size = 4, hidden_dim = 300, fc_dim = 200 , dropout = 0.5, n_layers=2)
model.load_state_dict(torch.load("./mman/state_dict/full/full73.14.pt", map_location=device));
esd_model = MLP(input_feature_size=33, hidden_size=118, n_classes=4,
                n_layers=1, device=device, p=0.2554070776341251)
model_info = torch.load("./saved_models/audio_mlp/ESD/4/4_model_ESD_acc_80.33k=0.a", map_location=device)
esd_model.load_state_dict(model_info["model_state_dict"])

threshold_angry = 0.3883602625489576
threshold_happy = 0.27169159422253325
threshold_sad = 0.43443322340931767
threshold_neutral = 0.24342304794959294

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

acc_test, f1_test, conf_matrix, classify_report = report_acc(output_baseline, torch.Tensor(text_test_label).to(device), text_test_mask)
print(classify_report)