#!/usr/bin/env bash
IMAGE_PATH=/home/thilini/neuraltalk2/coco/images/train2014
python3 obj_det_prep.py ${IMAGE_PATH} result_dict_coco.json