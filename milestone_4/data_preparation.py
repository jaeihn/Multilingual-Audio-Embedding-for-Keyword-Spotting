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

# https://huggingface.co/datasets/speech_commands

print("\n----------- MSWC Pytorch Dataset Prep -----------")

# Parsing arguments from command-line 

parser = argparse.ArgumentParser("MSWC Pytorch Dataset Prep")

parser.add_argument('csv_name', type=str, nargs=1, help='Name of .csv file containing data splits')

parser.add_argument('-mswc', '--mswc_dir', type=str, nargs='?', help='Path to the MSWC directory')
parser.add_argument('-data', '--data_dir', type=str, nargs='?', help='Path to the  directory containing keyword .txt files')

parser.add_argument('--train_batch', type=int, nargs='?', help='Train batch size (default == 1024)')
parser.add_argument('--dev_batch', type=int, nargs='?', help='Dev batch size (default == 1024)')
parser.add_argument('--test_batch', type=int, nargs='?', help='Test batch size (default == 1024)')
parser.add_argument('--max_samples', type=int, nargs='?', help='Max number of samples per keyword (default == 1000)')
# parser.add_argument('--model', type=str, nargs=1, help='One of efficientNet, whisper, wav2vec2')
parser.add_argument('-v', "--overwrite", action='store_true', help='Option to overwrite existing files')

args = parser.parse_args()
args = vars(args)

OVERWRITE = args['overwrite']
CSV_NAME = args['csv_name'][0]
MAX_SAMPLES = 1000

# MODEL = args['model'][0]

if args['max_samples']:
    MAX_SAMPLES = args['max_samples'][0]

# Batch sizes (Adjust default here)
TRAIN_BATCH = 8
DEV_BATCH = 256
TEST_BATCH = 256

if args['train_batch']:
    TRAIN_BATCH = args['train_batch'][0]
if args['dev_batch']:
    DEV_BATCH = args['dev_batch'][0]
if args['test_batch']:
    TEST_BATCH = args['test_batch'][0]


# Default paths
if args['mswc_dir']:
    PATH_MSWC = args['mswc_dir'].strip()
else:
    PATH_MSWC = '../../mswc'

if args['data_dir']:
    PATH_DATA = args['data_dir'].strip()
else: 
    PATH_DATA = '../data'
    
      
PATH_CSV = PATH_DATA.strip('/') + '/' + CSV_NAME

print(f"Building Datasets from File ... {PATH_CSV}")
print()

# Custom Functions  
    
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

        
# Build Vocabulary 


df_splits = pd.read_csv(PATH_CSV, header=0)
word_counter = Counter(df_splits.WORD)
vocab = word_counter.keys()

df_samples = []
for word in word_counter:
    if word_counter[word] > MAX_SAMPLES:
        df_samples.append(df_splits[df_splits.WORD==word].sample(MAX_SAMPLES, replace=False))

df_splits = pd.concat(df_samples, ignore_index=True)

WORD2IDX = {word: idx for idx, word in enumerate(vocab)}
WORD2IDX["<BGD>"] = len(WORD2IDX)

IDX2WORD = {idx: word for idx, word in enumerate(vocab)}
IDX2WORD[len(IDX2WORD)] = "<BGD>"

# Export word2idx 
with open("../data/idx2word_en_"+ str(len(WORD2IDX))+".csv", "w") as f:
    for k, v in IDX2WORD.items():
        f.write(str(k) + "," + str(v) + "\n")

# Build train, dev, test splits 

train_in, train_out = [], []
dev_in, dev_out = [], []
test_in, test_out = [], []

print(f"Preparing total of {len(df_splits)} data samples (before noise)... ") 

with tqdm(total=len(df_splits)) as pbar:    
    for i, row in df_splits.iterrows():
        LANG = row[1].split('common_voice_')[-1].split('_')[0]
        if row[0] == "TRAIN":
            train_in.append(get_audio_features(PATH_MSWC + '/' + LANG + '/clips/' + row[1]))
            train_out.append(torch.tensor(WORD2IDX.get(row[2], 0)))
        elif row[0] == "DEV":
            dev_in.append(get_audio_features(PATH_MSWC + '/' + LANG + '/clips/' + row[1]))
            dev_out.append(torch.tensor(WORD2IDX.get(row[2], 0)))
        elif row[0] == "TEST":
            test_in.append(get_audio_features(PATH_MSWC + '/' + LANG + '/clips/' + row[1]))
            test_out.append(torch.tensor(WORD2IDX.get(row[2], 0)))
        pbar.update(1)

# Add noise data, about 10% of dataset 
silence_dataset = [example for example in dataset['train'] if example['label'] == 35]

for split_in, split_out in [(train_in, train_out), (dev_in, dev_out), (test_in, test_out)]:
    n = int(len(split_in) * 0.1)
    # For the inputs, add about 10% of dataset noise samples
    samples = [get_audio_features(noise) for noise in generate_random_noise_samples(n, silence_dataset)]
    split_in += samples 
    # For the output, add "<BGD>" tag
    split_out += [torch.tensor(WORD2IDX["<BGD>"])] * n

print("Number of data points in each split")
print(f"\tTrain set : {len(train_in)}")
print(f"\tDev set : {len(dev_in)}")
print(f"\tTest set : {len(test_in)}")

# Convert to Dataset, Dataloader objects
train_dataset = KWS_dataset(train_in, train_out)
dev_dataset = KWS_dataset(dev_in, dev_out)
test_dataset = KWS_dataset(test_in, test_out)

torch.save(train_dataset, PATH_CSV[:-4] + '.train_dataset')
torch.save(dev_dataset, PATH_CSV[:-4] + '.dev_dataset')
torch.save(test_dataset, PATH_CSV[:-4] + '.test_dataset')

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=DEV_BATCH, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH, shuffle=True)

# Export for later use
torch.save(train_loader, PATH_CSV[:-4] + '.trainloader')
torch.save(dev_loader, PATH_CSV[:-4] + '.devloader')
torch.save(test_loader, PATH_CSV[:-4] + '.testloader')
