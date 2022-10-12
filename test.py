import pandas as pd
import numpy as np
import pickle
from utils import *


def main():
	_, _, audio_train, audio_test, text_train, text_test, video_train, video_test, train_label, test_label, seqlen_train, seqlen_test, train_mask, test_mask = get_iemocap_data(4)

if __name__ == '__main__':
    main()