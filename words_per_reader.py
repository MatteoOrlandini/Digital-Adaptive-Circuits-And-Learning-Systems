import json
import os


f = open("readers_paths.json", "r")

readers = json.load(f)
f.close()

for reader in readers:
    reader['words']=[{'word':"frar",'start':[1,2],'end':[3,4],'path':["dadsa","cacdc"]}]
    for path in reader['paths']:
        print(path)
        if (os.path.exists(path + "/word_count.json")):
            print("path esiste!")
            g = open(path + "/word_count.json", "r")
            words=json.load(g)
            g.close()
            for word in words:
                #print("parola")
                for x in reader['words']:
                    print("sono dentro l'array")
                    if word['word'] not in x['word']:
                        print("la parola è nuova")
                        reader['words'].append({'word':word['word'],'start':word['start'],'end':word['end'],'path':[path]*word['frequency']})
                    else:
                        print("la parola non è nuova")
                        for y in reader['words']:
                            for item in y['word']:
                                if word['word'] == item:
                                    item['start']+=word['start']
                                    item['end']+=word['end']
                                    item['path']+=[path]*word['frequency']

f=open("readers_words.json","w")
f.write(json.dumps(readers, indent = 4, sort_keys = False))
f.close()

