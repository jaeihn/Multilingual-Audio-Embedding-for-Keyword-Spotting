import argparse
import csv 
import librosa
import os
import pandas as pd
import torch
from collections import Counter
from datasets import load_dataset
from random import randint
from tqdm import tqdm
from transformers import WhisperFeatureExtractor
from torch.utils.data import DataLoader, Dataset

dataset = load_dataset("speech_commands", "v0.02")

print("\n----------- Keyword Spotting Dataset/Dataloader Prep -----------")

parser = argparse.ArgumentParser("Keyword Spotting Preparation")


# Parsing arguments from command-line 
parser.add_argument('csv_path', type=str, nargs=1, help='Path to .csv file containing data splits, relative to MSWC path')
parser.add_argument('txt_name', type=str, nargs=1, help='Name of .txt file containing embedding keywords')
parser.add_argument('keyword', type=str, nargs=1, help='Target keyword')

parser.add_argument('-mswc', '--mswc_dir', type=str, nargs='?', help='Path to the MSWC directory')
parser.add_argument('-data', '--data_dir', type=str, nargs='?', help='Path to the  directory containing keyword .txt files')

parser.add_argument('--train_batch', type=int, nargs='?', help='Train batch size (default == 256)')
parser.add_argument('--dev_batch', type=int, nargs='?', help='Dev batch size (default == 256)')
parser.add_argument('--test_batch', type=int, nargs='?', help='Test batch size (default == 256)')
parser.add_argument('--num_samples', type=int, nargs=1, help='Number of samples per keyword (default==256)')
parser.add_argument('-v', "--overwrite", action='store_true', help='Option to overwrite existing files')

args = parser.parse_args()
args = vars(args)

OVERWRITE = args['overwrite']
CSV_PATH = args['csv_path'][0]
TXT_NAME = args['txt_name'][0]
NUM_SAMPLES = 128
TARGET = args['keyword'][0]

# MODEL = args['model'][0]

if args['num_samples']:
    NUM_SAMPLES = args['num_samples'][0]

# Batch sizes (Adjust default here)
TRAIN_BATCH = int(NUM_SAMPLES / 0.4)
DEV_BATCH = int(NUM_SAMPLES / 0.4)
TEST_BATCH = int(NUM_SAMPLES / 0.4)

if args['train_batch']:
    TRAIN_BATCH = args['train_batch'][0]
if args['dev_batch']:
    DEV_BATCH = args['dev_batch'][0]
if args['test_batch']:
    TEST_BATCH = args['test_batch'][0]


# Default paths
if args['mswc_dir']:
    PATH_MSWC = args['mswc_dir'].strip('/')
else:
    PATH_MSWC = '../../mswc'

if args['data_dir']:
    PATH_DATA = args['data_dir'].strip('/')
else: 
    PATH_DATA = '../data'
    
PATH_CSV = PATH_MSWC+ '/' + CSV_PATH
PATH_TXT = PATH_DATA+ '/' + TXT_NAME 

print(f"Building Datasets from File ... {PATH_CSV}")
print()


# CUSTOM FUNCTIONS
def get_audio_features(file):
    '''This function takes in a file path or nd.array and returns extracted audio features using WhisperFeatureExtractor
    '''
    if type(file)==str:
        audio, sr = librosa.load(file) # load audio file
    else:
        audio = file
        
    feature_extractor = WhisperFeatureExtractor(feature_size=80, return_tensor='pt') # initialize feature extractor
    features = feature_extractor(audio, sampling_rate=16000) # extract features
    spectrogram = features['input_features'][0]
    spectrogram = spectrogram[:, :100] # crop spectrogram
    return spectrogram


def sample_noise(example):
    '''This function generates 1 sec random noises from the Speech Commands dataset in HuggingFace'''
    random_offset = randint(0, len(example['audio']['array']) - example['audio']['sampling_rate'] - 1)
    return example['audio']['array'][random_offset : random_offset + example['audio']['sampling_rate']]


def generate_random_noise_samples(num, silence_datasets):
    '''
    This function generates given number of 1 sec random noise samples 
    num : total number of samples wanted
    silence_datasets : list of silence datasets to pull from
    '''
    noise_samples = []
    # Populate `noise_samples` until we reach the desired number of samples 
    while len(noise_samples) < num:
        # Choose random dataset 
        random_dataset = randint(0,4)
        noise_samples.append(sample_noise(silence_datasets[random_dataset]))
    return noise_samples

# Custom Dataset 

class KWS_dataset(Dataset):
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, index):
        keyword = self.output_data[index]
        audio_features = self.input_data[index]
        return {'audio': audio_features, 'keyword': keyword}

    
    
# Load embedding words

with open(PATH_TXT) as f:
    IGNORE_WORDS = [w.strip() for w in f.readlines()]
IGNORE_WORDS.append(TARGET)


# Start target/non-target keyword extraction 
df = pd.read_csv(PATH_CSV, delimiter=',')

if not df.WORD.str.contains(TARGET).any():
    print("TARGET not found in CSV file !!!")
    exit()
    
# df_target = df[df.WORD==TARGET].sample(NUM_SAMPLES, replace=False)
# df_nontarget = df[~df.WORD.isin(IGNORE_WORDS)].sample(NUM_SAMPLES, replace=False)

# df_splits = pd.concat([df_target, df_nontarget])

train_in, train_out = [], []
dev_in, dev_out = [], []
test_in, test_out = [], []

#print(f"Preparing total of {len(df_splits)} data samples (before noise)... ") 

print("Prepare target words in TRAIN")
# TRAIN SET
df_target = df[(df.WORD==TARGET) & (df.SET=="TRAIN")].sample(NUM_SAMPLES, replace=False)

with tqdm(total=len(df_target)) as pbar:    
    for i, row in tqdm(df_target.iterrows()):
        LANG = row[1].split('common_voice_')[-1].split('_')[0]
        train_in.append(get_audio_features(PATH_MSWC + '/' + LANG + '/clips/' + row[1]))
        train_out.append(torch.tensor(0))
        pbar.update(1)

        
df_nontarget = df[(~df.WORD.isin(IGNORE_WORDS)) & (df.SET=="TRAIN")].sample(NUM_SAMPLES, replace=False)

with tqdm(total=len(df_nontarget)) as pbar:    
    for i, row in tqdm(df_nontarget.iterrows()):
        LANG = row[1].split('common_voice_')[-1].split('_')[0]
        train_in.append(get_audio_features(PATH_MSWC + '/' + LANG + '/clips/' + row[1]))
        train_out.append(torch.tensor(1))
        pbar.update(1)

print("Prepare DEV set (target/non-target words)")
# DEV SET 
dev_fraction = int(NUM_SAMPLES / 4.5)
df_target = df[(df.WORD==TARGET) & (df.SET=="DEV")].sample(dev_fraction, replace=False)

with tqdm(total=len(df_target)) as pbar:    
    for i, row in df_target.iterrows():
        LANG = row[1].split('common_voice_')[-1].split('_')[0]
        dev_in.append(get_audio_features(PATH_MSWC + '/' + LANG + '/clips/' + row[1]))
        dev_out.append(torch.tensor(0))
        pbar.update(1)

df_nontarget = df[(~df.WORD.isin(IGNORE_WORDS)) & (df.SET=="DEV")].sample(dev_fraction, replace=False)

with tqdm(total=len(df_nontarget)) as pbar:    
    for i, row in df_nontarget.iterrows():
        LANG = row[1].split('common_voice_')[-1].split('_')[0]
        dev_in.append(get_audio_features(PATH_MSWC + '/' + LANG + '/clips/' + row[1]))
        dev_out.append(torch.tensor(1))
        pbar.update(1)

    # elif row[0] == "TEST":
    #     test_in.append(get_audio_features(PATH_MSWC + '/' + LANG + '/clips/' + row[1]))
    #     test_out.append(torch.tensor(WORD2IDX.get(row[2], 0)))
    # pbar.update(1)

        
# Add noise data, about 10% of dataset 
silence_dataset = [example for example in dataset['train'] if example['label'] == 35]

for split_in, split_out in tqdm([(train_in, train_out), (dev_in, dev_out)]):#, (test_in, test_out)]):
    n = int(len(split_in) / 9) # == 10% of data
    # For the inputs, add about 10% of dataset noise samples
    samples = [get_audio_features(noise) for noise in generate_random_noise_samples(n, silence_dataset)]
    split_in += samples 
    # For the output, add "<BGD>" tag
    split_out += [torch.tensor(0)] * n

    
print("Number of data points in each split (after adding noise)")
print(f"\tTrain set : {len(train_in)}")
print(f"\tDev set : {len(dev_in)}")
# print(f"\tTest set : {len(test_in)}")


# Convert to Dataset, Dataloader objects
train_dataset = KWS_dataset(train_in, train_out)
dev_dataset = KWS_dataset(dev_in, dev_out)
# test_dataset = KWS_dataset(test_in, test_out)
 
torch.save(train_dataset, f"{PATH_DATA}/{TARGET}_{NUM_SAMPLES}_kws.test_dataset")
torch.save(dev_dataset, f"{PATH_DATA}/{TARGET}_{NUM_SAMPLES}_kws.test_dataset")
# torch.save(test_dataset, f"{PATH_DATA}/{TARGET}_{NUM_SAMPLES}_kws.test_dataset")

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=DEV_BATCH, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH, shuffle=True)

# Export for later use
torch.save(train_loader, f"{PATH_DATA}/{TARGET}_{NUM_SAMPLES}_kws.trainloader")
torch.save(dev_loader, f"{PATH_DATA}/{TARGET}_{NUM_SAMPLES}_kws.devloader")
# torch.save(test_loader, f"{PATH_DATA}/{TARGET}_{NUM_SAMPLES}_kws.testloader")
