import argparse
import os
import pandas as pd 
from collections import Counter

print("\n========== MSWC Vocabulary Ranker + Exporter ==========\n")

# Parsing arguments from command-line 

parser = argparse.ArgumentParser("MSWC Keyword Ranker")
parser.add_argument('input_dir', type=str, nargs=1, help='Path to the input (MSCW) directory')
parser.add_argument('language', type=str, nargs=1, help='Language')
parser.add_argument('output_dir', type=str, nargs=1, help='Path to the output (data) directory')
parser.add_argument('num_keywords', type=int, nargs=1, help='Number of keywords to export')
parser.add_argument('-o', "--overwrite", action='store_true', help='overwrite existing files')

args = parser.parse_args()
args = vars(args)

OVERWRITE = args['overwrite']
LANG = args['language'][0].lower()
NUM_KEYWORDS = args['num_keywords'][0]
PATH_CSV = args['input_dir'][0].strip('/') + '/' + LANG + '/'+ LANG + '_splits.csv'
PATH_TXT = args['output_dir'][0].strip('/') + '/' + 'keywords' + '_' + LANG + '_' + str(NUM_KEYWORDS) + '.txt'
    
# Read in .csv 

df_splits = pd.read_csv(PATH_CSV, usecols=['WORD'])

# Count and rank vocabulary 

vocab_ranking = sorted(Counter(df_splits.WORD).items(), key=lambda x: x[1], reverse=True)

# Export vocabulary list 

if os.path.exists(PATH_TXT):
    while True: 
        overwrite = input("File with same name already exists!! Overwrite? (y/n) ")
        if overwrite.startswith('y') or OVERWRITE:        
            with open(PATH_TXT, 'w') as f:
                f.writelines(line[0] + '\n' for line in vocab_ranking[:NUM_KEYWORDS])

            print(f"\nCreated vocabulary list for {LANG.upper()} with {NUM_KEYWORDS} keywords")
            print(f"\tFile generated at : {PATH_TXT}")
            print("====================== END ======================")
            break
        elif overwrite.startswith('n'):
            print("!! Aborted !!")
            print("====================== END ======================")
            break
else: 
    with open(PATH_TXT, 'w') as f:
        f.writelines(line[0] + '\n' for line in vocab_ranking[:NUM_KEYWORDS])

    print(f"\nCreated vocabulary list for {LANG.upper()} with {NUM_KEYWORDS} keywords")
    print(f"\tFile generated at : {PATH_TXT}")
    print("====================== END ======================")