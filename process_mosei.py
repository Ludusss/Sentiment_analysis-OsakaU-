import sys
sys.path.insert(1, '/Users/ludus/Projects/Sentiment_analysis-OsakaU-/CMU-MultimodalSDK')

from mmsdk import mmdatasdk
from mmsdk.mmdatasdk import log


def process_data(folders=["cmumosei_highlevel","cmumosei_labels"]):
    log.status("You can also download all the outputs of this code from here: http://immortal.multicomp.cs.cmu.edu/ACL20Challenge/")

    cmumosei_dataset={}
    for folder in folders:
        cmumosei_dataset[folder.split("_")[1]]=mmdatasdk.mmdataset(folder)
    print(cmumosei_dataset)


def download_data():
    source={"raw":mmdatasdk.cmu_mosei.raw,"highlevel":mmdatasdk.cmu_mosei.highlevel,"labels":mmdatasdk.cmu_mosei.labels}
    cmumosei_dataset={}
    print(source)
    for key in source:
        if key == "labels":
            cmumosei_dataset[key]=mmdatasdk.mmdataset(source[key],'cmumosei_%s/'%key)
    return cmumosei_dataset
