import pandas as pd
import torch
import numpy as np
from utils import *
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, './multimodal-emotion-recognition')
import model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Load Dataset
    data_train, data_test, audio_train, audio_test, text_train, text_test, video_train, video_test, train_label, test_label, seqlen_train, seqlen_test, train_mask, test_mask = get_iemocap_data(4)
    
    # Define model




if __name__ == '__main__':
    main()
