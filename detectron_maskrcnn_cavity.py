from pprint import pprint
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import cv2

# import some common detectron2 utilities
from detectron2.utils.visualizer import Visualizer

from detectron2.structures import BoxMode
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
import random
from detectron2.utils.visualizer import ColorMode

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg


def get_defect_anno_dict_train(train_annotations_path, train_dataset_path):
    print('GETTING DEFECT TRAIN ANNOTATION DICT')
    anno_dict_list = list()
    with open(train_annotations_path, 'r') as f:
        n_defects = 0  # count the total number of defects in training images
        anno = json.load(f)
        basic_image_data = anno['images']
        num_train_imgs = len(basic_image_data)

        annotation_image_data = anno['annotations']
        category_image_data = anno['categories']

        def modify_bbox(initial_bbox):
            bbox1 = initial_bbox[0]
            bbox2 = initial_bbox[1]
            bbox3 = bbox1 + initial_bbox[2]
            bbox4 = bbox2 + initial_bbox[3]
            bbox = [bbox1, bbox2, bbox3, bbox4]
            return bbox

        # Loop over image number and find objects in each image
        for img in range(num_train_imgs):
            filenames = list()
            heights = list()
            widths = list()
            iscrowds = list()
            bboxes = list()
            segmentations = list()
            bbox_modes = list()
            category_ids = list()
            # Get min image_id
            image_ids = list()
            for obj in annotation_image_data:
                image_ids.append(obj['image_id'])
            min_image_id = min(image_ids)
            if min_image_id == 0:
                shift = 0
            elif min_image_id == 1:
                shift = 1
            else:
                print('ERROR: minimum image_id should be 0 or 1')
            # Original data was separated by object, not by image
            for obj in annotation_image_data:
                if obj['image_id'] == img+shift:
                    filenames.append(basic_image_data[img]['file_name'])
                    heights.append(basic_image_data[img]['height'])
                    widths.append(basic_image_data[img]['width'])
                    iscrowds.append(obj['iscrowd'])
                    bboxes.append(modify_bbox(obj['bbox']))  # If doing XYXY_ABS bboxes
                    if obj['category_id'] == 1:  # cavity
                        category_ids.append(0)
                    #category_ids.append(1) #only cavities
                    segmentations.append(obj['segmentation'])
                    bbox_modes.append(BoxMode.XYXY_ABS)  # what I had before to follow balloon dataset
                    n_defects += 1

            anno_dict = dict()
            anno_dict['annotations'] = list()
            for iscrowd, bbox, category_id, segmentation, bbox_mode in zip(iscrowds, bboxes, category_ids,
                                                                           segmentations, bbox_modes):
                anno_dict['annotations'].append(
                    {'iscrowd': iscrowd, 'bbox': bbox, 'category_id': category_id, 'segmentation': segmentation,
                     'bbox_mode': bbox_mode})
            anno_dict['file_name'] = os.path.join(train_dataset_path, basic_image_data[img]['file_name'])
            anno_dict['height'] = basic_image_data[img]['height']
            anno_dict['width'] = basic_image_data[img]['width']
            anno_dict_list.append(anno_dict)

    print('Number of defects in training images')
    print(n_defects)
    #print('First train entry')
    #pprint(anno_dict_list[0])
    return anno_dict_list

def get_defect_anno_dict_val(test_annotations_path, test_dataset_path):
    print('GETTING DEFECT VAL ANNOTATION DICT')
    anno_dict_list = list()
    with open(test_annotations_path, 'r') as f:
        n_defects = 0  # count the total number of defects in val images
        anno = json.load(f)
        basic_image_data = anno['images']
        num_test_imgs = len(basic_image_data)
        annotation_image_data = anno['annotations']

        def modify_bbox(initial_bbox):
            bbox1 = initial_bbox[0]
            bbox2 = initial_bbox[1]
            bbox3 = bbox1 + initial_bbox[2]
            bbox4 = bbox2 + initial_bbox[3]
            bbox = [bbox1, bbox2, bbox3, bbox4]
            return bbox

        # Loop over image number and find objects in each image
        for img in range(num_test_imgs):
            filenames = list()
            heights = list()
            widths = list()
            iscrowds = list()
            bboxes = list()
            segmentations = list()
            bbox_modes = list()
            category_ids = list()
            # Original data was separated by object, not by image
            for obj in annotation_image_data:
                if obj['image_id'] == img:
                    filenames.append(basic_image_data[img]['file_name'])
                    heights.append(basic_image_data[img]['height'])
                    widths.append(basic_image_data[img]['width'])
                    iscrowds.append(obj['iscrowd'])
                    bboxes.append(modify_bbox(obj['bbox']))  # If doing XYXY_ABS bboxes
                    if obj['category_id'] == 1:  # cavity
                        category_ids.append(0)
                    #category_ids.append(1) #only cavities
                    segmentations.append(obj['segmentation'])
                    bbox_modes.append(BoxMode.XYXY_ABS)  # what I had before to follow balloon dataset
                    n_defects += 1

            anno_dict = dict()
            anno_dict['annotations'] = list()
            for iscrowd, bbox, category_id, segmentation, bbox_mode in zip(iscrowds, bboxes, category_ids,
                                                                           segmentations, bbox_modes):
                anno_dict['annotations'].append(
                    {'iscrowd': iscrowd, 'bbox': bbox, 'category_id': category_id, 'segmentation': segmentation,
                     'bbox_mode': bbox_mode})
            anno_dict['file_name'] = os.path.join(test_dataset_path, basic_image_data[img]['file_name'])
            anno_dict['height'] = basic_image_data[img]['height']
            anno_dict['width'] = basic_image_data[img]['width']
            anno_dict_list.append(anno_dict)
    print('Number of defects in val images')
    print(n_defects)
    #print('First val entry')
    #pprint(anno_dict_list[0])
    return anno_dict_list

def get_defect_metadata(train_dataset_path, test_dataset_path, train_annotations_path, test_annotations_path):
    # Attach the image metadata for class labels to the images
    DatasetCatalog.register(train_dataset_path, lambda : get_defect_anno_dict_train(train_annotations_path, train_dataset_path))
    MetadataCatalog.get(train_dataset_path).set(thing_classes=["void"], thing_colors=[(0,0,255)]) # only one class: cavity
    defect_metadata = MetadataCatalog.get(train_dataset_path)
    DatasetCatalog.register(test_dataset_path, lambda : get_defect_anno_dict_val(test_annotations_path, test_dataset_path))
    MetadataCatalog.get(test_dataset_path).set(thing_classes=["void"], thing_colors=[(0,0,255)]) # only one class: cavity
    print('Defect metadata')
    pprint(defect_metadata.as_dict())
    return defect_metadata

def visualize_image(anno_dict_list):
    # Get random image and show it
    for anno_dict in random.sample(anno_dict_list, 1):
    #for anno_dict in anno_dict_list:
        # Just set as first image to be reproducible:
        anno_dict = anno_dict_list[0]
        print('Visualizing image')
        print(anno_dict["file_name"])
        img = cv2.imread(anno_dict["file_name"])
        print('Image shape')
        print(img.shape)

        # Assign color to each defect in the image
        assigned_colors_list = list()
        for defect in anno_dict['annotations']:
            id = defect['category_id']
            if id == 0: #bdot
                assigned_colors_list.append('b')
            elif id == 1: #111
                assigned_colors_list.append('r')
            else:
                assigned_colors_list.append('y')
        anno_dict['assigned_colors'] = assigned_colors_list

        cv2_imshow(img)
        visualizer = Visualizer(img[:, :, ::-1], metadata=defect_metadata, scale=0.7, instance_mode=ColorMode.SEGMENTATION)
        vis = visualizer.draw_dataset_dict(anno_dict)
        cv2_imshow(vis.get_image()[:, :, ::-1])
        return

def get_config_file(input_yaml):
    cfg = get_cfg()
    if input_yaml['mask_on'] == True:
        if input_yaml['cascade_maskrcnn'] == True:
            cfg.merge_from_file("/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/Misc/" + str(input_yaml['pretrained_model_name']))
        else:
            cfg.merge_from_file("/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/"+str(input_yaml['pretrained_model_name']))
    elif input_yaml['mask_on'] == False:
        cfg.merge_from_file("/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-Detection/" + str(input_yaml['pretrained_model_name']))
    else:
        # Assume doing Mask R-CNN
        cfg.merge_from_file("/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/"+str(input_yaml['pretrained_model_name']))

    # Note that you can download the model weights from links provided on https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md
    model_weights_urls = {"cascade_mask_rcnn_R_50_FPN_1x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/Misc/model_weights/cascade_mask_rcnn_R_50_FPN_1x.pkl",
                          "cascade_mask_rcnn_R_50_FPN_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/Misc/model_weights/cascade_mask_rcnn_R_50_FPN_3x.pkl",

                        "mask_rcnn_R_50_C4_1x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/mask_rcnn_R_50_C4_1x_model_final_9243eb.pkl",
                        "mask_rcnn_R_50_C4_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/mask_rcnn_R_50_C4_3x_model_final_4ce675.pkl",
                        "mask_rcnn_R_50_DC5_1x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/mask_rcnn_R_50_DC5_1x_model_final_4f86c3.pkl",
                        "mask_rcnn_R_50_DC5_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/mask_rcnn_R_50_DC5_3x_model_final_84107b.pkl",
                        "mask_rcnn_R_50_FPN_1x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/mask_rcnn_R_50_FPN_1x_model_final_a54504.pkl",
                        "mask_rcnn_R_50_FPN_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/mask_rcnn_R_50_FPN_3x_model_final_f10217.pkl",
                        "mask_rcnn_R_50_FPN_3x_balloon": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/mask_rcnn_R_50_FPN_3x_balloon_model_final.pth",
                        "mask_rcnn_R_101_C4_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/mask_rcnn_R_101_C4_3x_model_final_a2914c.pkl",
                        "mask_rcnn_R_101_DC5_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/mask_rcnn_R_101_DC5_3x_model_final_0464b7.pkl",
                        "mask_rcnn_R_101_FPN_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/mask_rcnn_R_101_FPN_3x_model_final_a3ec72.pkl",
                        "mask_rcnn_X_101_32x8d_FPN_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/mask_rcnn_X_101_32x8d_FPN_3x_model_final_2d9806.pkl",

                          "faster_rcnn_R_50_C4_1x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/faster_rcnn_R_50_C4_1x_model_final_721ade.pkl",
                          "faster_rcnn_R_50_C4_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/faster_rcnn_R_50_C4_3x_model_final_f97cb7.pkl",
                          "faster_rcnn_R_50_DC5_1x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/faster_rcnn_R_50_DC5_1x_model_final_51d356.pkl",
                          "faster_rcnn_R_50_DC5_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/faster_rcnn_R_50_DC5_3x_model_final_68d202.pkl",
                          "faster_rcnn_R_50_FPN_1x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/faster_rcnn_R_50_FPN_1x_model_final_b275ba.pkl",
                          "faster_rcnn_R_50_FPN_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/faster_rcnn_R_50_FPN_3x_model_final_280758.pkl",
                          "faster_rcnn_R_101_C4_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/faster_rcnn_R_101_C4_3x_model_final_298dad.pkl",
                          "faster_rcnn_R_101_DC5_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/faster_rcnn_R_101_DC5_3x_model_final_3e0943.pkl",
                          "faster_rcnn_R_101_FPN_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/faster_rcnn_R_101_FPN_3x_model_final_f6e8b1.pkl",
                          "faster_rcnn_X_101_32x8d_FPN_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/faster_rcnn_X_101_32x8d_FPN_3x_model_final_68b088.pkl"
                          }

    if input_yaml['use_pretrained_model_weights'] == True:
        cfg.MODEL.WEIGHTS = model_weights_urls[input_yaml['pretrained_model_weights']]
        # Otherwise will use ImageNet weights instead of model-specific weights on CoCo data

    cfg.DATASETS.TRAIN = (input_yaml['train_dataset_path'],)
    cfg.DATASETS.TEST = (input_yaml['test_dataset_path'],)

    cfg.DATALOADER.NUM_WORKERS = 4  # Default was 2. Try 4?

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = input_yaml['max_iter']

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 class: void

    cfg.OUTPUT_DIR = input_yaml['output_dir']

    # HYPERPARAMS TO TUNE
    cfg.MODEL.RPN.IOU_THRESHOLDS = [input_yaml['rpn_iou_min'], input_yaml['rpn_iou_max']]  # Default RPN IoU thresholds. Anchors with overlap in between (BG_IOU_THRESHOLD <= IoU < FG_IOU_THRESHOLD)
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [input_yaml['roi_iou_threshold']]

    # Turn on input image augmentation (via cropping and I think image flipping)
    cfg.INPUT.CROP.ENABLED = input_yaml['crop_enabled']
    cfg.INPUT.CROP.SIZE = input_yaml['crop_size']

    # Adjust size of input training images
    #cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    cfg.INPUT.MIN_SIZE_TRAIN = tuple(input_yaml['input_min_size_train'])

    # Tune anchor generator sizes, aspect ratios and angles
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [input_yaml['anchor_angles']]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [input_yaml['anchor_sizes']]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [input_yaml['anchor_aspect_ratios']]

    # Number of layers to freeze backbone network
    cfg.MODEL.BACKBONE.FREEZE_AT = input_yaml['num_frozen_layers']

    # Choose between Mask R-CNN (mask is on) or Faster R-CNN (mask is off)
    cfg.MODEL.MASK_ON = input_yaml['mask_on']

    pprint(cfg)
    return cfg

def make_trainer(cfg, starting_fresh=True):
    if starting_fresh == True:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)  # only make output dir if starting fresh
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)  # starting fresh
    else:
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=True)  # starting from model saved in cfg.OUTPUT_DIR (default "output")
    return trainer

def run_trainer(trainer):
    trainer.train()
    return

# This is the main function to call to train maskrcnn from detectron2
def train_detectron_maskrcnn(input_yaml):
    try:
        defect_metadata = get_defect_metadata(input_yaml['train_dataset_path'],
                                              input_yaml['test_dataset_path'],
                                              input_yaml['train_annotations_path'],
                                              input_yaml['test_annotations_path'])
    except AssertionError:
        print(
            'Defect metadata has already been assigned. If you wish to reset the defect metadata, restart the runtime')
        pass

    cfg = get_config_file(input_yaml)
    trainer = make_trainer(cfg=cfg, starting_fresh=input_yaml['starting_fresh'])

    # Run if want to train model
    run_trainer(trainer=trainer)

    return cfg, defect_metadata



