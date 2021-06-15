import xml.etree.ElementTree as ET
import operator
import os
import collections
import pandas as pd
import json

words = []
target_words = []
# target_words = [{
    # 'word': [],
    # 'frequency': 0,
    # 'start' : []
# }]

folder = []
source_path = "./Dataset/English spoken wikipedia/english/"
filename = "aligned.swc"
f = open("word_count.json", "w")
for path in os.scandir(source_path):
    if (os.path.exists(path.path + "/" + filename)):
        tree = ET.parse(path.path + "/" + filename)
        # getroot returns the root element for this tree
        root = tree.getroot()
        # root.iter creates a tree iterator with the current element as the root. The iterator iterates over this element and all elements below it, in document (depth first) order.
        for token_normalization in root.iter(tag = 'n'):
            if 'start' in token_normalization.keys():
                # add every work with "start" key
                words.append(token_normalization.attrib['pronunciation'].lower())
        # collections.Counter stores elements as dictionary keys, and their counts are stored as dictionary values.
        unique_words = collections.Counter(words)
        #print("unique_words", unique_words)
        #print("unique_words length:", len(unique_words))
        for key in unique_words.keys():
            # we only consider words that occur at least 10 times in the recording
            if (unique_words[key] > 9):
                target_words.append({'word':key, 'frequency': unique_words[key], 'start':[], 'end':[]})
                
        for token_normalization in root.iter(tag = 'n'):
            if 'start' in token_normalization.keys():
                w = token_normalization.attrib['pronunciation'].lower()
                for target_word in target_words:
                    if w == target_word["word"]:
                        target_word["start"].append(int(token_normalization.attrib['start']))
                        target_word["end"].append(int(token_normalization.attrib['end']))


        f.write(path.path+ json.dumps(target_words, indent = 4, sort_keys = False))
        print("-")

f.close()

    # df=pd.DataFrame.from_dict(target_words)
    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    
    # print("target_words:", target_words)
    # print("target_words length:", len(target_words))

    # print("\n")
    # print(df)
    
    
    #for word in unique:
        #if word.values
    #unique.keys
            # if not any(item['word'] == token_normalization.attrib['pronunciation'].lower() for item in target_words):
                # dict = {'word' : token_normalization.attrib['pronunciation'], \
                        # 'frequency' : 1, \
                        # 'start' : [int(token_normalization.attrib['start'])]}
                # target_words.append(dict)
            # else:
                # for item in target_words:
                    # if (item['word'] == token_normalization.attrib['pronunciation'].lower()):
                        # item['frequency'] = item['frequency'] + 1
                        # item['start'].append(int(token_normalization.attrib['start']))
                        
    # print(target_words)
