import xml.etree.ElementTree as ET
import operator
import os
import collections
import pandas as pd
import json

# target_words = [{
    # 'word': [],
    # 'frequency': 0,
    # 'start' : [],
	# 'end' : []
# }]

folder = []
source_path = "./Dataset/English spoken wikipedia/english/"
filename = "aligned.swc"
# iterate in each folder of the dataset
for folder in os.scandir(source_path):
    enough_words = False
    if (os.path.exists(folder.path + "/" + filename)):  
        # parse the xml file aligned.swc
        tree = ET.parse(folder.path + "/" + filename)
        # getroot returns the root element for this tree
        root = tree.getroot()
        # initialize the list "words"
        words = []
        # root.iter creates a tree iterator with the current element as the root. The iterator iterates over this element and # all elements below it, in document (depth first) order.
        for token_normalization in root.iter(tag = 'n'):
            # we take only the words with the timestamp, so only if there is 'start' (or 'end') tag
            if 'start' in token_normalization.keys():
                # add every word with "start" key
                words.append(token_normalization.attrib['pronunciation'].lower())
        # collections.Counter stores elements as dictionary keys, and their counts are stored as dictionary values.
        unique_words = collections.Counter(words)
        # initialize the list of dict "target_words"
        target_words = []
        # for each key (word) in "unique_words" append a new target_word if the number of occurency is at least 10
        for key in unique_words.keys():
            # we only consider words that occur at least 10 times in the recording
            if (unique_words[key] >= 10):
                enough_words = True
                # add a new target word
                target_words.append({'word' : key, \
                                     'frequency' : unique_words[key], \
                                     'start' : [], \
                                     'end' : []})
        # for each "target_words" append the relative "start" and "end" timestamp
        for token_normalization in root.iter(tag = 'n'):
            # we take only the words with the timestamp, so only if there is 'start' (or 'end') tag
            if 'start' in token_normalization.keys():
                # iterate over the "target_words"
                for target_word in target_words:
                    # add start and end timestamp only to the relative "target_word"
                    if target_word['word'] == token_normalization.attrib['pronunciation'].lower():
                        target_word['start'].append(int(token_normalization.attrib['start']))
                        target_word['end'].append(int(token_normalization.attrib['end']))

    # save the "word_count.json" only if there are enough words 
    if (enough_words):
        f = open(folder.path+"/word_count.json", "w")
        f.write(json.dumps(target_words, indent = 4, sort_keys = False))
        print("Folder:", folder.path)
        f.close()
