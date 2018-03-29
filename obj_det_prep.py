#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 12:44:04 2018

@author: 1002636
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Mask_RCNN import object_detection

import json
from collections import OrderedDict

# content_image_regions = json.load(open('region_descriptions_updated_final.json'), object_pairs_hook=OrderedDict)
#img_dir1 = '/Users/thilinicooray/whatever/VG_100K'
#img_dir2 = '/Users/thilinicooray/whatever/VG_100K_2'
batch_size = 10

obj_det = object_detection.OBG_det(batch_size)

image_id_list = []
image_path_list = []
# for image in content_image_regions:
#     image_id = image['id']
#     image_id_list.append(image_id)
#     img_dir = '/home/thilini/sem-img/new_work/Dense_VSRL/images'
#     ''' if os.path.exists(os.path.join(img_dir1, str(image_id)+'.jpg')):
#         img_dir = img_dir1
#     else:
#         img_dir = img_dir2'''
#     image_path_list.append(os.path.join(img_dir, str(image_id)+'.jpg'))

image_dir = sys.argv[1]

from pathlib import Path
image_dir_path = Path(image_dir)
image_lst = image_dir_path.glob('*.jpg')
image_path_list = [str(x.absolute()) for x in image_lst]
temp = [Path(x).name for x in image_path_list]
image_id_list = [x[:x.find('.')] for x in temp]

print('Total image size = ', len(image_id_list))
img_count = 0

results_dict = {}
while img_count < len(image_path_list):
    print('CURRENT IMG COUNT :',img_count, len(image_path_list))
    if len(image_path_list) - img_count < batch_size:
        new_batch_size = len(image_path_list) - img_count
        obj_det = object_detection.OBG_det(new_batch_size)
        end = len(image_path_list)
    else:
        end = img_count + batch_size
    print('END :', end)
    try:
        result_list = obj_det.detect_objects(image_path_list[img_count:end])
    except:
        print ('Error at', image_id_list[img_count:end])
        img_count = end
        continue

    print('results :', result_list)
    for i in range(img_count, end):
        print('counter = ', i)
        results_dict[image_id_list[i]] = result_list[i%batch_size]
    img_count = end

print('OBJ detection for all images completed.')

result_file_path = sys.argv[2]
with open (result_file_path, 'w') as outputfile:
    json.dump(results_dict, outputfile)

