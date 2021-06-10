import xml.etree.ElementTree as ET
import operator
import os
import json

# Initialize list with empty dictionaries
readers = [] 
# readers = [{
	# 'reader_name': [],
	# 'frequency': 0,
	# 'paths' : []
# }]

folder = []
source_path = "./Dataset/English spoken wikipedia/english/"
filename = "aligned.swc"

for audio_path in os.scandir(source_path):
	if (os.path.exists(audio_path.path + "/" + filename)):
		tree = ET.parse(audio_path.path + "/" + filename)
		# getroot returns the root element for this tree
		root = tree.getroot()
		# root.iter creates a tree iterator with the current element as the root. The iterator iterates over this element and all elements below it, in document (depth first) order.
		for property in root.iter(tag = 'prop'):
			if (property.attrib['key'] == 'reader.name'):
				if not any(reader['reader_name'] == property.attrib['value'].lower() for reader in readers):
					dict = {'reader_name': property.attrib['value'].lower(), \
							'frequency': 1, \
							'paths' : [audio_path.path] \
						}
					readers.append(dict)
				else:
					for reader in readers:
						if (reader['reader_name'] == property.attrib['value'].lower()):
							reader['frequency'] = reader['frequency'] + 1
							reader['paths'].append(audio_path.path)
		
print("len(readers): ", str(len(readers)))

# popularoutcast
f = open("readers_paths.txt", "w")
f.write(json.dumps(readers, indent = 4, sort_keys = False))
f.close()

	#for item in target_words:
	#	print(item)