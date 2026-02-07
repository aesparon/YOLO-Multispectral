


import sys
import os
#import shutil
import numpy as np
#from PIL import Image
from pathlib import Path
#import yaml
import pandas as pd
from collections import Counter

#import torch

# code folder
#yolo_source_path = 'D:/PD/Publications/Yolo_mod/github/YOLO-Multispectral/code/'
#sys.path.insert(1, yolo_source_path)

#from mod_pt_model_seg_v15 import *
#from mod_pt_model_seg_v40 import *

#  NOTE if use_source = True will use source - NOT package ULTRLYTICS in python environment
#from ultralytics import YOLO






def create_train_yaml(images_train_path, images_val_path ,classes_list , number_of_channels, output_yaml ,  images_test_path = None):



    ###    CREATE data - YAML file for training    #######################################################################
    yaml_content = "train: " + images_train_path + "\n"
    yaml_content =yaml_content +  "val: " + images_val_path + "\n"
    if not (images_test_path == '' or images_test_path == None):
        yaml_content =yaml_content +  "test: " + images_test_path + "\n"

    # extra channels
    yaml_content =yaml_content +  "nc: " + str ( len(classes_list) ) + "\n"
    yaml_content =yaml_content +  "channels: " + str( number_of_channels) + "\n"
    #yaml_content =yaml_content  + cat_yaml_str     #"names: ['tree']"

    #classes_list = ['maize','amaranth','grass','quickweed','other']  
    # create name classes for yolo training
    names_str = "names: [" + ",".join(classes_list) + "]"
    print(names_str)
    yaml_content =yaml_content +  names_str + "\n"

    #output_yaml =  project_base_path + "data_" + train_data + ".yaml"
    

    with Path(output_yaml).open('w') as f:
        f.write(yaml_content) 
    ##########################################################################################################################





def eval_best_last(model_save_path , output_yaml , best_last , yolo_source_path ):
    sys.path.insert(0, yolo_source_path)
    from ultralytics import YOLO

    #  if output_model_train folder already exists model.train will create a new folder with incremented postfix
    #model_save_path = str(results.save_dir)

    # output test results
    test_base_out = model_save_path + '/test_' + best_last + '/'   
    
    model =  model_save_path + '/weights/' + best_last + '.pt'
    model_path =  model   # 'D:/PD/yolo_mod/working2/projects/weeds_galore/RGB_tests/weed_RGB_yolo8x-seg_no_TL_eps_200/weights/' + 'best.pt'
    model = YOLO(model_path)  # or continue using your model object
    results_test = model.val(data=output_yaml, split='test' , name = test_base_out)

    out_csv = test_base_out + 'out.csv'
    

    names = results_test.names  # class_id -> name
    box = results_test.box
    seg = results_test.seg
    nc = len(names)

    # Get instance counts from AP class index
    instance_counter = Counter(box.ap_class_index)
    instance_counts = [instance_counter.get(i, 0) for i in range(nc)]

    # Image count is not tracked directly, default to -1
    image_counts = [-1] * nc

    # Create dataframe
    df = pd.DataFrame({
        'class_id': list(names.keys()),
        'class_name': list(names.values()),
        'Instance_Count': instance_counts,
        'Image_Count': image_counts,

        'Box_Precision': box.p,
        'Box_Recall': box.r,
        'Box_mAP@0.5': box.ap50,
        'Box_mAP@0.5:0.95': box.ap,

        'Mask_Precision': seg.p,
        'Mask_Recall': seg.r,
        'Mask_mAP@0.5': seg.ap50,
        'Mask_mAP@0.5:0.95': seg.ap
    })

    # Mean row
    mean_row = {
        'class_id': '-',
        'class_name': 'mean (all classes)',
        'Instance_Count': sum(instance_counts),
        'Image_Count': sum(i if i != -1 else 0 for i in image_counts),

        'Box_Precision': box.mp,
        'Box_Recall': box.mr,
        'Box_mAP@0.5': box.map50,
        'Box_mAP@0.5:0.95': box.map,
        'Mask_Precision': seg.mp,
        'Mask_Recall': seg.mr,
        'Mask_mAP@0.5': seg.map50,
        'Mask_mAP@0.5:0.95': seg.map
    }
    df.loc[len(df)] = mean_row

    # Save
    df.to_csv(out_csv, index=False)
    