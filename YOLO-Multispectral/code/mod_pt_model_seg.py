import os
import shutil
import sys
import copy
import shutil
from pathlib import Path

import torch
import torch.nn as nn

# # path
# yolo_source_path = 'D:/PD/yolo_mod/working2/yolo_source/yolo_2025_06_04_mod/ultralytics-main/'
# sys.path.insert(1, yolo_source_path)
# import ultralytics
# from ultralytics import YOLO
# from ultralytics.nn.modules.conv import Conv
# #from ultralytics.nn.tasks import SegmentationModel
# from ultralytics.cfg import get_cfg
# # Load no_TL model with trusted unpickling
# from ultralytics.nn.tasks import SegmentationModel








def patch_yolo_seg_ckpt(model_base, output_model_train , yolo_source_path , in_channels=5, nc=5 , channel_init_mode='avg' , use_cbam=False, use_eca=False, use_spectral=False, use_dropblock=False, drop_prob=0.1):
    print(f"\n📦 Loading pretrained YOLO segmentation model: {model_base}")


    # path to modified yolo source code for multispectral training
    #yolo_source_path = 'D:/PD/yolo_mod/working2/yolo_source/yolo_2025_06_04_mod/ultralytics-main/'
    sys.path.insert(1, yolo_source_path)
    #import ultralytics
    from ultralytics import YOLO
    from ultralytics.nn.modules.conv import Conv
    #from ultralytics.nn.tasks import SegmentationModel
    from ultralytics.cfg import get_cfg
    # Load no_TL model with trusted unpickling
    from ultralytics.nn.tasks import SegmentationModel

    from patch_backbone_with_attention import patch_backbone_with_attention


    ####   UPDATE later just pass   yolov11x-seg.yaml    OR    yolov11x-seg.pt


    # append extra modifications for tracebility
    if use_cbam:
        output_model_train = output_model_train + '_use_cbam'
    if use_eca:
        output_model_train = output_model_train + '_use_eca'
    if use_spectral:
        output_model_train = output_model_train + '_use_spectral'
    if use_dropblock:
        output_model_train = output_model_train + '_use_dropblock'


    model_base_wo_ext, extension = os.path.splitext(model_base)




    #  Check mask or BB and qppend to file path for tracebility
    if model_base_wo_ext[-3:] == 'seg':
        bb_seg = 'seg'
    else:
        bb_seg = 'bb'

    
    # if extension == '.pt':
    #     #  pt model with weights -> trans_learn:
    #     #pre-trainined
    #     #input_ckpt =  model_base + '.pt' 

    #     # new modified model
    #     output_model_train = output_model_train  + '_TL_'  + bb_seg 
    #     model_created = output_model_train + '_mod/'  + model_base +  '_TL.pt'

    # else:
    #     # get yaml from 
    #     #input_ckpt = model_base + '.yaml'

    #     # new modified model
    #     output_model_train = output_model_train  + '_no_TL_'  + bb_seg 
    #     model_created = output_model_train + '_mod/' + model_base +  '_no_TL.pt'



    # Tidy later
    input_ckpt = model_base




    if extension == '.yaml':

        # load yaml file
        model = YOLO(input_ckpt)
        # model = YOLO(model_yaml_path)
        # # Save the model to a .pt file
        save_yaml_ckpt =  model_base_wo_ext + '_yaml.pt'
        model.save(save_yaml_ckpt)

        no_tl_patch =  model_base_wo_ext + '_yaml_patch.pt'
        # with torch.serialization.safe_globals([SegmentationModel]):
        #     ckpt = torch.load(save_yaml_ckpt , map_location='cpu', weights_only=False)
        ckpt = torch.load(save_yaml_ckpt, map_location='cpu')
        
        # Extract raw model
        raw_model = ckpt['model']

        # Rebuild new checkpoint format 
        new_ckpt = {
            'model': raw_model,
            'epoch': 0,
            'best_fitness': 0.0,
            'ema': None,
            'updates': 0,
            'optimizer': None,
            'train_args': {
                'task': 'segment',          # maybe only upadte mod required
                #'imgsz': 640,
                #'batch': 16,
                #'epochs': 300,
                #'model': input_ckpt ,
                #'data': 'data.yaml',
                #'resume': False,
                #'device': 'cuda:0',
            },
            'date': ckpt.get('date', ''),
            'version': ckpt.get('version', ''),
            'license': ckpt.get('license', ''),
            'docs': ckpt.get('docs', '')
        }

        # Save new TL-style .pt file
        torch.save(new_ckpt, no_tl_patch)

        model = YOLO(no_tl_patch)

        # new modified model from YAML so no transfer learning
        output_model_train = output_model_train  + '_no_TL_'  + bb_seg 
        model_created = output_model_train + '/mod_model/' + model_base_wo_ext +  '_no_TL.pt'

        
    elif extension == '.pt' or extension == '':
        # if no extension default to .pt  
        model = YOLO(input_ckpt)
        #  pt model with weights -> trans_learn:
        # add for traceility
        output_model_train = output_model_train  + '_TL_'  + bb_seg 
        model_created = output_model_train + '/mod_model/'  + model_base_wo_ext  +  '_TL.pt'
    else:
        # update 
        raise Exception("base_model should have extension .pt or .yaml - if no extension defaults to .pt!")


    os.makedirs(os.path.dirname(model_created), exist_ok=True)

    # Get actual nn.Module
    model_nn = model.model  # This is usually a nn.Sequential or custom model

    # Access first Conv layer
    if isinstance(model_nn, nn.Sequential):
        conv = model_nn[0].conv
    else:
        conv = model_nn.model[0].conv

    print(f"🛠️ Patching input conv from 3 to {in_channels} using '{channel_init_mode}'")
    weight = conv.weight.data

    if in_channels > 3:
        new_channels = in_channels - 3
        if channel_init_mode == 'avg':
            avg = weight[:, :3, :, :].mean(dim=1, keepdim=True)
            extra = avg.expand(-1, new_channels, -1, -1)
        elif channel_init_mode == 'zeros':
            extra = torch.zeros(weight.size(0), new_channels, *weight.shape[2:], device=weight.device)
        elif channel_init_mode == 'random':
            std = weight.std().item()
            extra = torch.randn(weight.size(0), new_channels, *weight.shape[2:], device=weight.device) * std
        elif channel_init_mode == 'same':
            extra = weight[:, :new_channels, :, :].clone()
        else:
            raise ValueError(f"Unknown channel_init_mode: {channel_init_mode}")
        conv.weight = nn.Parameter(torch.cat([weight, extra], dim=1))

    elif in_channels < 3:
        conv.weight = nn.Parameter(weight[:, :in_channels, :, :])

    conv.in_channels = in_channels

    # Update segmentation head
    print(f"🧠 Updating segmentation head to {nc} classes")
    seg_head = model_nn[-1] if isinstance(model_nn, nn.Sequential) else model_nn.model[-1]
    seg_head.nc = nc
    if hasattr(seg_head, 'predictor'):
        seg_head.predictor[-1] = nn.Conv2d(seg_head.predictor[-1].in_channels, nc, kernel_size=1)



    # patch any additional modifications
    if use_cbam or use_eca or use_spectral or use_dropblock:
        patch_backbone_with_attention(model.model, use_cbam=use_cbam, use_eca=use_eca , use_spectral =use_spectral , use_dropblock = use_dropblock , drop_prob = drop_prob )


    print(f"Saving patched model to: {model_created}")
    model.save(model_created)

    return model_created ,output_model_train
