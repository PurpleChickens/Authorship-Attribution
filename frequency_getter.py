import numpy as np
import json
import nltk
import glob
import os
word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

# load data
data_folder = r"corpus"
files = sorted(glob.glob(os.path.join(data_folder, "*")))
books = []
for fn in files:
    with open(fn) as f:
        books.append(f.read().replace('\n', ''))

''' Returns frequency of function words '''
def get_func_word_freq(words,funct_words):
    fdist = nltk.FreqDist(x for x in words if x in funct_words) 
    funct_freq = {}
    for key,value in fdist.items():
        funct_freq[key] = value
    for key,value in funct_freq.items():
        funct_freq[key] = value/float(len(words))
    return funct_freq

''' Returns a MxN array where M = #books, N = #function words and element 
    (m,n) is the frequency of the nth function word of the mth book '''
def calc_func_word_freq():
    funct_words = load_func_word()
    freq_array = np.zeros((len(books), len(funct_words)), np.float64)
    for e, text in enumerate(books):
        words = word_tokenizer.tokenize(text.lower())
        x = get_func_word_freq(words,funct_words)
        for i,value in enumerate(funct_words):
            if value in x:
                freq_array[e,i] = x[value]
    return freq_array 

''' Returns a list of all the function words '''
def load_func_word():
    data_folder = r"English_Function_Words_Set"
    files = glob.glob(os.path.join(data_folder, "*.txt"))
    function_words = []
    for fn in files:
        with open(fn) as f:
            lines = f.readlines()
            for line in lines:
                if line[0] == '/':
                    continue;
                function_words.append(line.replace('\n', ''))
    return list(function_words)    
    
if __name__ == '__main__':
    for x,y in zip(files,calc_func_word_freq()):
        x = "freq_"+ x 
        f = open(x, 'w')
        json.dump(y.tolist(), f)
