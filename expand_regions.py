
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from visual_genome import api
from PIL import Image as PIL_Image
import requests
from io import StringIO, BytesIO
import sys
import json


# In[3]:




res_file_path = '/home/ta/Projects/SUTD/VSRL/data/VisualGenome/semantic_relationsips_final_full.json'
object_file_path = '/home/ta/Projects/SUTD/VSRL/data/VisualGenome/objects.json'
region_file_path = '/home/ta/Projects/SUTD/VSRL/data/VisualGenome/region_descriptions.json'

res_data = json.load(open(res_file_path))
object_data = json.load(open(object_file_path))
region_data = json.load(open(region_file_path))


# In[4]:


mapping_id = {
    'region': {},
    'object': {}
}

for id_, region in enumerate(region_data):
    mapping_id['region'][region['id']] = id_

for id_, object in enumerate(object_data):
    mapping_id['object'][object['image_id']] = id_


# In[5]:


from collections import namedtuple

Rec = namedtuple('Rec', 'xmin ymin xmax ymax')

def intersect(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
        return 0.0

def get_verlap_ratio(region1,region2,obj):
    obj_region_intsct = intersect(region1, region2)
#     print('overlapping area : ', obj_region_intsct)

    if obj_region_intsct == 0.0:
        return 0.0

    object_overlap_ratio = obj_region_intsct / float(obj['w']*obj['h'])
#     print('overlap ratio:', object_overlap_ratio)
    return object_overlap_ratio


# In[6]:


def visualize_regions_objects(image, regions, objects, new_region=None):
    fig = plt.gcf()
    # fig.set_size_inches(18.5, 10.5)
    fig.set_size_inches(30, 16, forward=True)
    response = requests.get(image.url)
    img = PIL_Image.open(BytesIO(response.content))
    
    ax = plt.gca()
#     f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#     ax1.imshow(img)
#     old
    plt.imshow(img)

    if regions != None:
        for region in regions:
            ax.add_patch(Rectangle((region['x'], region['y']),
                                   region['width'],
                                   region['height'],
                                   fill=False,
                                   edgecolor='red',
                                   linewidth=3))
            ax.text(region['x'],
                    region['y'],
                    region['phrase'] + ': ' + str(region['region_id']),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})

    if objects != None:
        for object_ in objects:
            ax.add_patch(Rectangle((object_['x'], object_['y']),
                                   object_['w'],
                                   object_['h'],
                                   fill=False,
                                   edgecolor='blue',
                                   linewidth=3))

            ax.text(object_['x'],
                    object_['y'],
                    object_['names'][0] + ': ' + str(object_['object_id']),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
            
    if new_region != None:
#         print (new_region)
        ax.add_patch(Rectangle((new_region.xmin, new_region.ymin),
                               new_region.xmax - new_region.xmin,
                               new_region.ymax - new_region.ymin,
                                fill=False,
                               edgecolor='yellow',
                               linewidth=3))
    
        ax.text(new_region.xmin,
                new_region.ymin,
                "new_bbox",
                style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})

#     ax1 = ax 
    fig = plt.gcf()
#     old
    plt.tick_params(labelbottom='off', labelleft='off')

    if objects == None or len(objects) == 0:
#         plt.savefig('res/bbox_region/' + str(regions[0]['region_id']))
        pass 
    else:
#         print('Detect objects!')
#         print(objects)
        plt.savefig('res/expand_bbox_res/' + str(regions[0]['region_id']))
#         plt.show()
        plt.clf()
    


# In[7]:


import math 

for image in res_data:
    image_id = image['image_id']
    print ('Current processing at image', image_id)
    # graph_image = api.get_scene_graph_of_image(image_id)
    # objects = graph_image.objects
    objects = object_data[mapping_id['object'][image_id]]['objects']

    # regions = api.get_region_descriptions_of_image(id=image_id)
    regions = region_data[mapping_id['region'][image_id]]['regions']
    raw_image = api.get_image_data(id=image_id)

    for region in image['relationships']:
        objects_region = []
        for k, v in region['region_relations'].items():
            if isinstance(v,int):
                objects_region.append(v)
            else:
#                 print (v, type(v))
                numbers = [int(e) for e in v.split() if e.isdigit()]
                objects_region.extend(numbers)

        selected_regions = [x for x in regions if x['region_id'] == region['region_id']]
        selected_objects = [x for x in objects if x['object_id'] in objects_region]
        
#         print (selected_objects) 
#         print (selected_regions)
        
        error_image = False 
        
        region_bb = selected_regions[0] 
        region_box = Rec(region_bb['x'], 
                         region_bb['y'], 
                         region_bb['x'] + region_bb['width'], 
                         region_bb['y'] + region_bb['height'])
        
        xmin = region_bb['x']
        ymin = region_bb['y']
        xmax = region_bb['x'] + region_bb['width']
        ymax = region_bb['y'] + region_bb['height']
        
        for obj in selected_objects: 
#             print ('Current object:', obj['names'][0])
            xmin = min(obj['x'], xmin) 
            ymin = min(obj['y'], ymin) 
            xmax = max(obj['x'] + obj['w'], xmax) 
            ymax = max(obj['y'] + obj['h'], ymax) 
            
            obj_box = Rec(obj['x'], 
                          obj['y'], 
                          obj['x'] + obj['w'], 
                          obj['y'] + obj['h'])
#             print (region_box) 
#             print (obj_box)
            
            object_overlap_ratio = get_verlap_ratio(region_box, obj_box, obj)
            if object_overlap_ratio < 0.5: 
                error_image = True 
                break 
    
        if error_image:
            new_region_box = Rec(xmin - 2, ymin - 2, xmax + 2, ymax + 2) 
        else:
            new_region_box = None
            
        visualize_regions_objects(raw_image, selected_regions, selected_objects, new_region_box)
#         break 
#     break 
    print ('')

