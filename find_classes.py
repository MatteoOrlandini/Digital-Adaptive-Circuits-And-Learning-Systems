import xml.etree.ElementTree as ET
import operator
import os
import collections
import pandas as pd
import json
import random



# target_words = [{
    # 'word': [],
    # 'frequency': 0,
    # 'start' : [],
	# 'end' : []
# }]

folder = []
source_path = "./Dataset/English spoken wikipedia/english/"
filename = "aligned.swc"
for path in os.scandir(source_path):
    if (os.path.exists(path.path + "/" + filename)):    
        
        tree = ET.parse(path.path + "/" + filename)
        # getroot returns the root element for this tree
        root = tree.getroot()

        words = []
        # root.iter creates a tree iterator with the current element as the root. The iterator iterates over this element and # all elements below it, in document (depth first) order.
        for token_normalization in root.iter(tag = 'n'):
            if 'start' in token_normalization.keys():
                # add every work with "start" key
                words.append(token_normalization.attrib['pronunciation'].lower())
        # collections.Counter stores elements as dictionary keys, and their counts are stored as dictionary values.
        unique_words = collections.Counter(words)
        target_words = []
        for key in unique_words.keys():
            # we only consider words that occur at least 26 times in the recording
            if (unique_words[key] > 9):
                target_words.append({'word' : key, \
                                     'frequency' : unique_words[key], \
                                     'start' : [], \
                                     'end' : []})
        
        if len(target_words)>9:
            f = open(path.path+"/word_count.json", "w")
            #reduce the array of words in 10 words made of 26 samples
            
            #target_words = random.sample(target_words, 10)
            
            #for word in target_words:

            for token_normalization in root.iter(tag = 'n'):
                if 'start' in token_normalization.keys():
                    w = token_normalization.attrib['pronunciation'].lower()
                    for target_word in target_words:
                        if w == target_word['word']:
                            target_word['start'].append(int(token_normalization.attrib['start']))
                            target_word['end'].append(int(token_normalization.attrib['end']))

            

            f.write(json.dumps(target_words, indent = 4, sort_keys = False))
            print(path.path)
            f.close()
