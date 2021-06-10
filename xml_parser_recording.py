import xml.etree.ElementTree as ET
import operator
import os

target_words = [{
	'word': [],
	'frequency': 0,
	'start' : []
}]

folder = []
source_path = "./Dataset/English spoken wikipedia/english/"
filename = "aligned.swc"

for path in os.scandir(source_path):
	if (os.path.exists(path.path + "/" + filename)):
		tree = ET.parse(path.path + "/" + filename)
		# getroot returns the root element for this tree
		root = tree.getroot()
		# root.iter creates a tree iterator with the current element as the root. The iterator iterates over this element and all elements below it, in document (depth first) order.
		for token_normalization in root.iter(tag = 'n'):
			if 'start' in token_normalization.keys():
				if not any(item['word'] == token_normalization.attrib['pronunciation'].lower() for item in target_words):
					dict = {'word' : token_normalization.attrib['pronunciation'], \
							'frequency' : 1, \
							'start' : [int(token_normalization.attrib['start'])]}
					target_words.append(dict)
				else:
					for item in target_words:
						if (item['word'] == token_normalization.attrib['pronunciation'].lower()):
							item['frequency'] = item['frequency'] + 1
							item['start'].append(int(token_normalization.attrib['start']))

		#target_words.sort(key = operator.itemgetter('frequency'), reverse = False)

		### try ###
		
		# a = numpy.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4])
		# unique, counts = numpy.unique(a, return_counts = True)
		# dict(zip(unique, counts))
		
		# {0: 7, 1: 4, 2: 1, 3: 2, 4: 1}
		
		#### or #####
		
		# import collections, numpy
		# a = numpy.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4])
		# collections.Counter(a)

		# {0: 7, 1: 4, 2: 1, 3: 2, 4: 1}

		count = 0
		for item in target_words:
			if (item['frequency'] > 9):
				count = count + 1
				
		if (count > 9):
			folder.append(path.path)
		
		print(len(folder), path.path)
		
f = open("paths.txt", "w")
for path in folder:
	f.write(path)
f.close()

	#for item in target_words:
	#	print(item)