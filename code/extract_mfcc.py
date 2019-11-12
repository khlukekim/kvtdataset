from __future__ import print_function
import librosa
import numpy as np
import os
import pickle

hop = 512
melBin = 128
N_MFCC_C = 13

def main():
  extract_folder('segment_mix', 'mfcc_mix')

def extract_folder(input_path, output_path):
  files = os.listdir(input_path)
  for f in files:
    if f[-3:] in ['wav', 'mp3']:
      extract(os.path.join(input_path, f), os.path.join(output_path, f.split('.')[0]))

def extract(input_file, output_file):
  y, sr = librosa.load(input_file, sr=22050)
  mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC_C, n_mels=melBin)
  mfcc = mfcc.astype('float32')
  np.save(output_file, mfcc)

if __name__ == '__main__':
  main()






