import json

def write_json_file(filename, list_of_dict, indent = 0):
    f = open(filename, "w")
    f.write(json.dumps(list_of_dict, indent = indent))
    f.close

def read_json_file(filename):
    f = open(filename, "r")
    list_of_dict = json.load(f)
    f.close
    return list_of_dict