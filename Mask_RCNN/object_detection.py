
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.

# In[89]:

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from . import coco
from . import utils
from . import model as modellib
from . import visualize

#get_ipython().magic('matplotlib inline')

class OBG_det():
    def __init__(self, batch_size):
        # Root directory of the project
          #ROOT_DIR = os.getcwd()
          ROOT_DIR = '/home/thilini/sem-img/new_work/new_data_filtering/Mask_RCNN'
        
          # Directory to save logs and trained model
          MODEL_DIR = os.path.join(ROOT_DIR, "logs")
        
          # Local path to trained weights file
          COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
          # Download COCO trained weights from Releases if needed
          if not os.path.exists(COCO_MODEL_PATH):
              utils.download_trained_weights(COCO_MODEL_PATH)
              
          #image_path_list = image_path_list[:4]
        
          # Directory of images to run detection on
          #IMAGE_DIR = os.path.join(ROOT_DIR, "images")
        
        
          # ## Configurations
          # 
          # We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
          # 
          # For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.
        
          # In[90]:
          #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        
          # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
          #sess.run()
          class InferenceConfig(coco.CocoConfig):
              # Set batch size to 1 since we'll be running inference on
              # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
              GPU_COUNT = 1
              IMAGES_PER_GPU = batch_size
        
          config = InferenceConfig()
          config.display()
        
        
          # ## Create Model and Load Trained Weights
        
          # In[91]:
        
          # Create model object in inference mode.
          self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
          #opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
          #conf = tf.ConfigProto(gpu_options=opts)
          #infConfig = tf.estimator.RunConfig(session_config=conf)
          #model =  tf.estimator.Estimator(model_fn=model.keras_model, config=infConfig)
          # Load weights trained on MS-COCO
          self.model.load_weights(COCO_MODEL_PATH, by_name=True)
        
        
          # ## Class Names
          # 
          # The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
          # 
          # To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
          # 
          # To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
          # ```
          # # Load COCO dataset
          # dataset = coco.CocoDataset()
          # dataset.load_coco(COCO_DIR, "train")
          # dataset.prepare()
          # 
          # # Print class names
          # print(dataset.class_names)
          # ```
          # 
          # We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)
        
          # In[92]:
        
    def detect_objects(self,image_path_list):
  
          # COCO Class names
          # Index of the class in the list is its ID. For example, to get ID of
          # the teddy bear class, use: class_names.index('teddy bear')
          class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                         'bus', 'train', 'truck', 'boat', 'traffic light',
                         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                         'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                         'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                         'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                         'kite', 'baseball bat', 'baseball glove', 'skateboard',
                         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                         'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                         'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                         'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                         'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                         'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                         'teddy bear', 'hair drier', 'toothbrush']
        
          living_being_ids = [1, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        
        
          # ## Run Object Detection
        
          # In[93]:
        
          # Load a random image from the images folder
          #file_names = next(os.walk(IMAGE_DIR))[2]
          #image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
          images = []
          for image_path in image_path_list:
            image = skimage.io.imread(image_path)
            #print('shapeeeee :', len(image.shape))
            if len(image.shape) != 3:
              image = skimage.color.gray2rgb(image)
            images.append(image)
        
          # Run detection
          results = self.model.detect(images, verbose=1)
          
          final_results = []
          
          for i in range(len(results)):
              detected_classes = results[i]['class_ids']
        
              print('detected object class IDs :', detected_classes)
              #living_beings = [x for x in detected_classes if x in living_being_ids]
              living_being_count = 0
              
              image_area = float(images[i].shape[0] * images[i].shape[1])
              
              for j in range(len(detected_classes)):
                  if detected_classes[j] in living_being_ids:
                      y1, x1, y2, x2 = results[i]['rois'][j]
                      object_bbox_area = float(x2-x1)*(y2-y1)
                      
                      coverage = float(object_bbox_area*100)/image_area
                      print('coverage : ', coverage, '%')
                      
                      if coverage >= 15.00:
                          living_being_count += 1
                  #roi_list.append()
              final_results.append(True if living_being_count > 2 else False)    
        
          # Visualize results
          #r = results[0]
          #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                      #class_names, r['scores'])
        
        
          # In[ ]:
          #sess.close()
          return final_results
    
