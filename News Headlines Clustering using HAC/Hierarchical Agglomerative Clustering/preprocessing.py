# Cleaning scrapped data to remove punctuation, numbers and make it to lower case
# Write the data to a file to be used further

import re
from string import punctuation
import os


def preprocessing():

    path = os.path.abspath(os.path.dirname(__file__))
    f = open(os.path.join(path, 'data\dataset_used.txt'), 'rt', encoding='utf-8')
    text_file = f.read().split('\n')

    text_To_lower = [text.lower() for text in text_file]
    letters = [''.join(c for c in s if c not in punctuation) for s in text_To_lower]

    final = [re.sub(r'[^A-Za-z]+', ' ', x) for x in letters]

    with open(os.path.join(path, 'data\dataset_cleaned.txt'), 'w') as fw:
        for text in final:
            fw.write(text)
            fw.write('\n')
    fw.close()


