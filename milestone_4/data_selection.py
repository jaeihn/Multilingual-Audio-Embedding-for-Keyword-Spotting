import argparse
import csv 
import os
import pandas as pd 
import nltk
from collections import Counter
from nltk.corpus import stopwords
from tqdm import tqdm 

print("\n----------- MSWC Keyword Selector -----------")

# Parsing arguments from command-line 

parser = argparse.ArgumentParser("MSWC Keyword Selector")

parser.add_argument('language', type=str, nargs=1, help='Language')
parser.add_argument('num_keywords', type=int, nargs=1, help='Number of keywords to export')

parser.add_argument('-mswc', '--mswc_dir', type=str, nargs='?', help='Path to the MSWC directory')
parser.add_argument('-kw', '--kw_dir', type=str, nargs='?', help='Path to the  directory containing keyword .txt files')
parser.add_argument('-v', "--overwrite", action='store_true', help='Option to overwrite existing files')

args = parser.parse_args()
args = vars(args)

OVERWRITE = args['overwrite']
LANG = args['language'][0].lower()
NUM_KEYWORDS = args['num_keywords'][0]

# Default paths

if args['mswc_dir']:
    PATH_INPUT = args['mswc_dir']
else:
    PATH_INPUT = '../../mswc'

if args['kw_dir']:
    PATH_OUTPUT = args['kw_dir']
else: 
    PATH_OUTPUT = '../data'
    
print(f"Input path : {PATH_INPUT}")
print(f"Output path : {PATH_OUTPUT}")
print()
      
PATH_CSV = PATH_INPUT.strip('/') + '/' + LANG + '/'+ LANG + '_splits.csv'
PATH_TXT = PATH_OUTPUT.strip('/') + '/' + 'keywords' + '_' + LANG + '_' + str(NUM_KEYWORDS) + '.txt'
PATH_CSV_SELECT = PATH_OUTPUT.strip('/') + '/' + LANG + '_splits_' + str(NUM_KEYWORDS) + '.csv'
    
    
# Read in original .csv 

df_splits = pd.read_csv(PATH_CSV, usecols=['WORD'])

# Count and rank vocabulary 

vocab_ranking = sorted(Counter(df_splits.WORD).items(), key=lambda x: x[1], reverse=True)

# Remove stopwords
vocab_ranking = [word for word in vocab_ranking if word[0] not in stopwords.words('english')]

# Export vocabulary list 

if os.path.exists(PATH_TXT):
    while True: 
        overwrite = input("File with same name already exists!! Overwrite? (y/n) ")
        if overwrite.startswith('y') or OVERWRITE:        
            with open(PATH_TXT, 'w') as f:
                f.writelines(line[0] + '\n' for line in vocab_ranking[:NUM_KEYWORDS])

            print(f"\nCreated vocabulary list for {LANG.upper()} with {NUM_KEYWORDS} keywords")
            print(f"\tFile generated at : {PATH_TXT}")
            print("\n============================================")
            break
        elif overwrite.startswith('n'):
            print("!! Aborted !!")
            print("Continuing to selection ... ")
            print("\n============================================")
            break
else: 
    with open(PATH_TXT, 'w') as f:
        f.writelines(line[0] + '\n' for line in vocab_ranking[:NUM_KEYWORDS])

    print(f"\nCreated vocabulary list for {LANG.upper()} with {NUM_KEYWORDS} keywords")
    print(f"\tFile generated at : {PATH_TXT}")
    print("\n============================================")
    
# Reload keywords 

with open(PATH_TXT, 'r') as f:
    vocab = f.read().split('\n')
    
vocab = [v for v in vocab if v != '']

# Count number of lines in file 

with open(PATH_CSV, 'r') as f:
    total_count = sum(1 for line in f)
    
count = 0

with open(PATH_CSV, 'r') as f_original, open(PATH_CSV_SELECT, 'w') as f_selected:
    reader = csv.reader(f_original)
    writer = csv.writer(f_selected, delimiter=',')

    # Copy header
    writer.writerow(next(reader))
    
    # Copy words in vocab
    with tqdm(total=total_count) as pbar:
        for row in reader:
            if row[2] in vocab:
                writer.writerow(row)
                count += 1
            pbar.update(1)
            
print(f"\nCreated .csv file with {count} lines for {NUM_KEYWORDS} keywords")
print(f"\tFile generated at : {PATH_CSV_SELECT}")
print("\n====================== END ======================")