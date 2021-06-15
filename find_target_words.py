import xml.etree.ElementTree as ET
import operator
import os
import collections

words = []
target_words = []
# target_words = [{
	# 'word': [],
	# 'frequency': 0,
	# 'start' : []
# }]

folder = []
source_path = "./Dataset/English spoken wikipedia/english/Asperger_syndrome/"
filename = "aligned.swc"

#for path in os.scandir(source_path):
if (os.path.exists(source_path + "/" + filename)):
	tree = ET.parse(source_path + "/" + filename)
    # getroot returns the root element for this tree
	root = tree.getroot()
    # root.iter creates a tree iterator with the current element as the root. The iterator iterates over this element and all elements below it, in document (depth first) order.
	for token_normalization in root.iter(tag = 'n'):
		if 'start' in token_normalization.keys():
			# add every work with "start" key
			words.append(token_normalization.attrib['pronunciation'].lower())
	# collections.Counter stores elements as dictionary keys, and their counts are stored as dictionary values.
	unique_words = collections.Counter(words)
	print("unique_words", unique_words)
	print("unique_words length:", len(unique_words))
	for key in unique_words.keys():
		# we only consider words that occur at least 10 times in the recording
		if (unique_words[key] > 9):
			target_words.append({'word':key, 'frequency': unique_words[key]})
			
	print("target_words:", target_words)
	print("target_words length:", len(target_words))
	
	
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
