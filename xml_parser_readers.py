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

source_path = "./Dataset/English spoken wikipedia with audio/english/"
filename = "aligned.swc"

for audio_path in os.scandir(source_path):
	if (os.path.exists(audio_path.path + "/" + filename)):
		# example: audio_path.path = ./Dataset/English spoken wikipedia with audio/english/Revolt_of_the_Admirals
		audio_path_split = audio_path.path.split('/')
		folder = audio_path_split[len(audio_path_split)-1]

		tree = ET.parse(audio_path.path + "/" + filename)
		# getroot returns the root element for this tree
		root = tree.getroot()
		# root.iter creates a tree iterator with the current element as the root. The iterator iterates over this element and all elements below it, in document (depth first) order.
		for property in root.iter(tag = 'prop'):
			if (property.attrib['key'] == 'reader.name'):
				reader_name = property.attrib['value'].lower()
				if ("user:" in reader_name):
					if ("|" in reader_name):
						# example reader_name = [[:en:user:alexkillby|alexkillby]] -> reader_name = alexkillby
						reader_name = reader_name[reader_name.find("user:")+5:reader_name.find("|")]
					elif ("]]" in reader_name):
						# example reader_name = [[user:popularoutcast]] -> reader_name = popularoutcast
						reader_name = reader_name[reader_name.find("user:")+5:reader_name.find("]]")]
					
				if not any(reader['reader_name'] == reader_name for reader in readers):
					dict = {'reader_name': reader_name, \
							'frequency': 1, \
							'folder' : [folder] \
						}
					readers.append(dict)
				else:
					for reader in readers:
						if (reader['reader_name'] == reader_name):
							reader['frequency'] = reader['frequency'] + 1
							reader['folder'].append(folder)
		
print("len(readers): ", str(len(readers)))

# popularoutcast
f = open("readers_paths.json", "w")
f.write(json.dumps(readers, indent = 4, sort_keys = False))
f.close()