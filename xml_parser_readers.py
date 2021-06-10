import xml.etree.ElementTree as ET
import operator
import os
import json

readers = [{
	'reader': [],
	'frequency': 0,
	'paths' : []
}]

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
				if not any(item['reader'] == property.attrib['value'].lower() for item in readers):
					dict = {'reader': property.attrib['value'].lower(), \
							'frequency': 1, \
							'paths' : [audio_path.path] \
						}
					readers.append(dict)
				else:
					for item in readers:
						if (item['reader'] == property.attrib['value'].lower()):
							item['frequency'] = item['frequency'] + 1
							item['paths'].append(audio_path.path)
		
print("len(readers): ", str(len(readers)))

f = open("readers_paths.txt", "w")
f.write(json.dumps(readers))
f.close()

	#for item in target_words:
	#	print(item)