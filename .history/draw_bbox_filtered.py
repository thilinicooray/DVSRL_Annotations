import sys
import os
import json

res_file_path = './result_dict_coco.json'
data = json.load(open(res_file_path))
for k, v in data.items():
    if v == True:
        print (k)

