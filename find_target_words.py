import xml.etree.ElementTree as ET
import operator
import os
import collections

words = []
target_words = [{
	'word': [],
	'frequency': 0,
	'start' : []
}]

folder = []
source_path = "../Progetti Github/Digital-Adaptive-Circuits-And-Learning-Systems/Dataset/English spoken wikipedia/english/Asperger_syndrome/"
filename = "aligned.swc"

#for path in os.scandir(source_path):
if (os.path.exists(source_path + "/" + filename)):
    tree = ET.parse(source_path + "/" + filename)
    # getroot returns the root element for this tree
    root = tree.getroot()
    # root.iter creates a tree iterator with the current element as the root. The iterator iterates over this element and all elements below it, in document (depth first) order.
    for token_normalization in root.iter(tag = 'n'):
        if 'start' in token_normalization.keys():
            words.append(token_normalization.attrib['pronunciation'].lower())
    unique = collections.Counter(words)
    print(unique)
    for word in unique:
        if word.values
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
