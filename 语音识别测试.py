# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：
时间：
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import IPython
import wave
from td_utils import *
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam

x = graph_spectrogram("/Users/simon/Mycodes/Speech-Rec/临时/BAC009S0150W0001.wav")

_, data = wavfile.read("/Users/simon/Mycodes/Speech-Rec/临时/BAC009S0150W0001.wav")
print("Time steps in audio recording before spectrogram", data.shape)
print("Time steps in input after spectrogram", x.shape)  # (101, 5511)

f = wave.open("/Users/simon/Mycodes/Speech-Rec/临时/BAC009S0150W0001.wav")
print(f)
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
print(params)


def main():
    pass


if __name__ == "__main__":
    main()
