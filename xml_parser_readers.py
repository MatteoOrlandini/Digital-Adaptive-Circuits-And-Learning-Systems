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

source_path = "./Dataset/English spoken wikipedia/english/"
filename = "aligned.swc"

# search for readers for each folder
for audio_path in os.scandir(source_path):
	# # save only folder name from entire path
	folder =  os.path.basename(audio_path)
	if (os.path.exists(source_path + "/" + folder + "/" + filename)):
		# parse the xml file "aligned.swc"
		tree = ET.parse(source_path + "/" + folder + "/" + filename)
		# getroot returns the root element for this tree
		root = tree.getroot()
		# root.iter creates a tree iterator with the current element as the root. The iterator iterates over this element and all elements below it, in document (depth first) order.
		for property in root.iter(tag = 'prop'):
			# if the key "reader.name" exists
			if (property.attrib['key'] == 'reader.name'):
				# save the reader name taking the value of the attribute
				reader_name = property.attrib['value'].lower()
				# fix readers names that contain "user:"
				if ("user:" in reader_name):
					# fix readers names that contain "|""
					if ("|" in reader_name):
						# example reader_name = [[:en:user:alexkillby|alexkillby]] -> reader_name = alexkillby
						reader_name = reader_name[reader_name.find("user:")+5:reader_name.find("|")]
					# fix readers names that contain "|""
					elif ("]]" in reader_name):
						# example reader_name = [[user:popularoutcast]] -> reader_name = popularoutcast
						reader_name = reader_name[reader_name.find("user:")+5:reader_name.find("]]")]
				# if the reader is not yet on the list create a dict and append to the readers list
				if not any(reader['reader_name'] == reader_name for reader in readers):
					dict = {'reader_name': reader_name, \
							'frequency': 1, \
							'folder' : [folder] \
						}
					readers.append(dict)
				else:
					# if the reader is already on the list add the folder name
					for reader in readers:
						if (reader['reader_name'] == reader_name):
							reader['frequency'] = reader['frequency'] + 1
							reader['folder'].append(folder)
# print the number of the readers
print("len(readers): ", str(len(readers)))

# save a "readers_paths.json" with the name of the readers and the relative file audio folders
f = open("readers_paths.json", "w")
f.write(json.dumps(readers, indent = 4, sort_keys = False))
f.close()