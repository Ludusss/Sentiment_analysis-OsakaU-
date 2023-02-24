import sys
import time

from luo.utils import report_acc, cal_acc, cal_acc_without_mask

sys.path.append('/Users/ludus/Projects/Sentiment_analysis-OsakaU-/luo')
from luo.model import LstmModel, hir_fullModel
from model import MLP
from utils import process_features, get_iemocap_data, report_acc_mlp
import numpy as np
import torch
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#_, _, _, _, _, _, _, _, _, audio_features_test, audio_labels_test, audio_mask_test, _, _ = process_features(True)
_, _, _, text_test, text_test_label, text_test_mask, _, _, _, audio_test, audio_test_label, audio_mask_test, _, _ = process_features(True)
text_test_mask = text_test_mask.reshape(3286)
#model = LstmModel(input_feat_size=33, output_size=4, hidden_dim=300, fc_dim=200, dropout=0.5)
model = hir_fullModel(batch_size = 10, mode = False, classifier = "lstm", output_size = 4, hidden_dim = 300, fc_dim = 200 , dropout = 0.5, n_layers=2)
esd_model = MLP(input_feature_size=33, hidden_size=118, n_classes=4,
                              n_layers=1, device=device, p=0.2554070776341251)
#model.load_state_dict(torch.load("./luo/state_dict/audioRNN/audioRNN50.19.pt", map_location=device))
#model.load_state_dict(torch.load("./luo/state_dict/bl_hira/bl_hira73.07.pt", map_location=device));
model.load_state_dict(torch.load("./luo/state_dict/full/full73.14.pt", map_location=device));

dir = "./saved_models/audio_mlp/ESD/4"
esd_models = os.listdir(dir)
best_acc = 0
best_model = ""
best_combined = 0
for model_text in esd_models:
    model_info = torch.load(dir + "/" + model_text, map_location=device)
    esd_model.load_state_dict(model_info["model_state_dict"])

    model.eval()
    esd_model.eval()
    with torch.no_grad():
        #input_data = torch.Tensor(np.array(audio_test))
        input_data_audio = torch.Tensor(audio_test)
        input_data_text = torch.Tensor(text_test)
        #output, _ = model(input_data)
        output, _ = model(input_data_audio, input_data_text)
        output = torch.softmax(output, dim=1)
        acc, misclassified_samples = cal_acc(output, text_test_label, text_test_mask)
        print("Accuracy of baseline: " + str(acc))

        input_data_esd = torch.Tensor(np.reshape(audio_test, (3286, 33)))
        input_labels_esd = torch.Tensor(np.reshape(audio_test_label, 3286))

        total_true = [x for x in audio_mask_test.reshape(-1) if x == 1]
        input_data_esd = torch.Tensor(np.take(input_data_esd, misclassified_samples, axis=0))
        input_labels_esd = np.reshape(audio_test_label, 3286)
        input_labels_esd = np.take(input_labels_esd, misclassified_samples, axis=0)

        output_esd = esd_model(input_data_esd)
        output_esd = torch.softmax(output_esd, dim=1)
        acc_mis = cal_acc_without_mask(output_esd, input_labels_esd)

        print("Accuracy of ASN on misclassified samples: " + str(acc_mis))

        if acc_mis > best_acc:
            best_model = model_text
            best_acc = acc_mis
            best_combined = acc + ((input_data_esd.shape[0]*best_acc)/len(total_true))

print("Best model was: " + best_model + " with accuracy: " + str(best_acc) + " combined accuracy: " + str(best_combined))
