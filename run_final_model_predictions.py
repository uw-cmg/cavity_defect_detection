import detectron2
from detectron2.utils.visualizer import Visualizer, ColorMode
import cv2, os
from google.colab.patches import cv2_imshow
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import pandas as pd
from shapely.geometry import Polygon as shapelyPolygon
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def visualize_image(IMAGE_PATH, MY_IMAGE):
    print('On image', MY_IMAGE)
    im = cv2.imread(os.path.join(IMAGE_PATH, MY_IMAGE))
    #cv2_imshow(im)
    return

def get_config(IMAGE_PATH, MODEL_PATH, SAVE_PATH, NUM_CLASSES, CLASS_NAMES, CLASS_COLORS):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = MODEL_PATH  # path to the model we just trained
    cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.4]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.OUTPUT_DIR = SAVE_PATH
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.DATASETS.TEST = (IMAGE_PATH,)
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [0.5, 1, 2]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [4, 8, 16, 32, 64, 128, 256, 512]
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25
    
    defect_metadata = MetadataCatalog.get(IMAGE_PATH).set(thing_classes=CLASS_NAMES, thing_colors=CLASS_COLORS)

    predictor = DefaultPredictor(cfg)

    return predictor, defect_metadata

def visualize_pred_image(IMAGE_PATH, SAVE_PATH, MY_IMAGE, predictor, defect_metadata):
    im = cv2.imread(os.path.join(IMAGE_PATH, MY_IMAGE))
    print(im.shape)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                  metadata=defect_metadata,
                  scale=1.0,
                  instance_mode=ColorMode.SEGMENTATION)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    im_preds = out.get_image()[:, :, ::-1]
    cv2_imshow(im_preds)
    ext = MY_IMAGE[-4:]
    cv2.imwrite(os.path.join(SAVE_PATH, MY_IMAGE[:-4]+'_withpreds'+ext), im_preds)
    return im, outputs

def get_pred_data(instances):
    pred_classes = list()
    pred_segmentations = list()
    pred_boxes = list()
    pred_scores = list()
    n_masks = 0

    for mask in instances.pred_masks:

        mask_id = list()
        defect_id = list()

        pred_boxes.append(np.array(instances.pred_boxes[n_masks].tensor.to("cpu")).tolist()[0])
        pred_classes.append(int(instances.pred_classes[n_masks].to("cpu")))

        pred_coords_x = np.where(mask.to("cpu"))[0].tolist()
        pred_coords_y = np.where(mask.to("cpu"))[1].tolist()

        for i in range(len(pred_coords_x)):
            mask_id.append(n_masks)
            defect_id.append(int(instances.pred_classes[n_masks].to("cpu")))

        points = np.array([pred_coords_y, pred_coords_x])
        points = points.T
        vertices = get_mask_vertices(points)
        vertices_y = list(vertices[:, 0])
        vertices_x = list(vertices[:, 1])
        vertices_y, vertices_x = (list(t) for t in zip(*sorted(zip(vertices_y, vertices_x))))
        vertices = np.array([vertices_y, vertices_x]).T
        pred_segmentations.append([vertices[:, 0].tolist(), vertices[:, 1].tolist()])
        
        pred_scores.append(float(instances.scores[n_masks].to("cpu")))

        n_masks += 1
        
    return pred_classes, pred_boxes, pred_segmentations, pred_scores

def get_true_data(anno, 
                        image_name, 
                        image_path,
                        save_path, 
                        defect_metadata,
                        visualize_image=True, 
                        save_images=True):
    # Find the right image
    found_image = False
    for i, anno_dict in enumerate(anno['images']):
        base_imagename = image_name.split('/')[-1]
        base_filename = anno_dict["file_name"].split('/')[-1]
        if base_filename == base_imagename:
            image_id = anno['images'][i]['id']
            found_image = True
            print('Successfully found image', image_name, 'with image id', image_id)
            break
    if found_image == False:
        print(
            'WARNING: An error occurred, the provided filename could not be corresponded with the name of an image file in the provided annotation dictionaries')
        return

    img = cv2.imread(os.path.join(image_path, anno_dict["file_name"]))

    # Need to get the list of annotations pertaining to this image
    anno_list = list()
    anno_dict = dict()
    for a in anno['annotations']:
        if a['image_id'] == image_id:
            a['bbox_mode'] = BoxMode.XYWH_ABS
            a['category_id'] = 0
            anno_list.append(a)
    anno_dict['annotations'] = anno_list
    print('Found true annotations', len(anno_list))

    print(anno_dict['annotations'])

    # Assign color to each defect in the image
    assigned_colors_list = list()
    for defect in anno_list:
        id = defect['category_id']
        if id == 0:  # bdot
            assigned_colors_list.append('b')
        elif id == 1:  # 111
            assigned_colors_list.append('r')
        else:
            assigned_colors_list.append('y')
    anno_dict['assigned_colors'] = assigned_colors_list

    if visualize_image == True:
        visualizer = Visualizer(img[:, :, ::-1],
                                metadata=defect_metadata,
                                scale=1.0,
                                instance_mode=ColorMode.SEGMENTATION
                                )

        vis = visualizer.draw_dataset_dict(anno_dict)
        img2 = vis.get_image()[:, :, ::-1]

        if save_images == True:
            cv2.imwrite(os.path.join(save_path, base_imagename + '_true.png'), img2)

    # PARSE AND OUTPUT TRUE PIXEL DATA
    df_list_true = list()
    true_classes = list()
    true_boxes = list()
    true_segmentations = list()
    n_masks = 0
    for obj in anno_dict['annotations']:
        data_dict_true = dict()
        seg_y = list()
        seg_x = list()
        seg_y_nearestint = list()
        seg_x_nearestint = list()
        true_classes.append(obj['category_id'])
        true_boxes.append(obj['bbox'])
        #true_segmentations = list()

        n_defects = len(anno_dict['annotations'])
        for i, seg in enumerate(obj['segmentation'][0]):
            if i == 0:
                seg_y.append(seg)
                seg_y_nearestint.append(int(seg))
            else:
                if i % 2 == 0:
                    seg_y.append(seg)
                    seg_y_nearestint.append(int(seg))
                else:
                    seg_x.append(seg)
                    seg_x_nearestint.append(int(seg))

        segmentation = np.array([seg_y, seg_x]).T
        segmentation_nearestint = np.array([seg_y_nearestint, seg_x_nearestint]).T
        data_dict_true['segmentation_y'] = segmentation[:, 0]
        data_dict_true['segmentation_x'] = segmentation[:, 1]
        true_segmentations.append([segmentation_nearestint[:, 0].tolist(), segmentation_nearestint[:, 1].tolist()])

    return true_classes, true_segmentations, true_boxes, img

def get_mask_vertices(points):
    try:
        hull = ConvexHull(points)
    except:
        # Usually an error occurs if the input is too low-dimensional due to rounding of pixel values. Joggle input to help avoid this
        hull = ConvexHull(points, qhull_options='QJ')
    vertices = np.array([points[hull.vertices, 0], points[hull.vertices, 1]]).T
    return vertices

def get_density(im, pred_classes, nm_per_pixel, num_classes):
    classes = np.unique(np.array(pred_classes))
    area = (nm_per_pixel * im.shape[0]) * (nm_per_pixel * im.shape[1])
    if num_classes == 1:
        density = 10**4*len(pred_classes) / area
        return density
    else:
        densities = list()
        for cls in classes:
            densities.append(10**4*len(np.array(pred_classes)[np.where(np.array(pred_classes) == cls)]) / area)
        return densities

def get_defect_size(segmentation, defect_type, nm_per_pixel):
    # Using a segmentation mask for a defect (true or pred), calculate the defect radius (in pixels)
    # Get center of the segmentation mask (note the array needs to be 2D)
    segmentation_xy = list()
    seg_x = segmentation[0]
    seg_y = segmentation[1]
    for x, y in zip(seg_x, seg_y):
        segmentation_xy.append([x, y])

    segmentation_xy = np.array(segmentation_xy)
    center = np.array([[np.mean(segmentation_xy[:, 0]), np.mean(segmentation_xy[:, 1])]])
    min_distance = min(cdist(segmentation_xy, center))
    min_distance = min_distance[0]
    max_distance = max(cdist(segmentation_xy, center))
    max_distance = max_distance[0]

    defect_radius_pixels = np.sqrt(min_distance * max_distance)
    defect_radius_nm = defect_radius_pixels * nm_per_pixel
    defect_diameter_nm = 2 * defect_radius_nm

    # Need to get defect shape factor. Using Heywood circularity factor, = perimeter / circumference of circle of same area
    # Need to use convex hull construction of segmentation mask, so points follow an ordered path, making the shape closed
    hull = ConvexHull(segmentation_xy)
    vertices = np.array([segmentation_xy[hull.vertices, 0], segmentation_xy[hull.vertices, 1]]).T
    polygon = shapelyPolygon(vertices)
    perimeter = polygon.length
    area = polygon.area
    radius = np.sqrt(area / np.pi)
    circumference = 2 * np.pi * radius
    defect_shape_factor = perimeter / circumference

    return defect_diameter_nm, defect_shape_factor

def get_sizes_and_shapes_image(pred_segmentations, pred_classes, nm_per_pixel, num_classes):
    pred_sizes = dict()
    pred_shapes = dict()
    if num_classes == 1:
        pred_sizes[0] = list()
        pred_shapes[0] = list()
    elif num_classes == 2:
        pred_sizes[0] = list()
        pred_shapes[0] = list()
        pred_sizes[1] = list()
        pred_shapes[1] = list()
    elif num_classes == 3:
        pred_sizes[0] = list()
        pred_shapes[0] = list()
        pred_sizes[1] = list()
        pred_shapes[1] = list()
        pred_sizes[2] = list()
        pred_shapes[2] = list()
    for seg, defect in zip(pred_segmentations, pred_classes):
        defect_size, defect_shape_factor = get_defect_size(segmentation=seg,
                                                           defect_type=defect,
                                                           nm_per_pixel=nm_per_pixel)
        pred_sizes[defect].append(defect_size)
        pred_shapes[defect].append(defect_shape_factor)
    if num_classes == 1:
        return pred_sizes[0], pred_shapes[0]
    else:
        return [pred_sizes[pred_size] for pred_size in list(pred_sizes.keys())], [pred_shapes[pred_shape] for pred_shape
                                                                                  in list(pred_shapes.keys())]

def get_swelling(im, image_thickness, nm_per_pixel, pred_sizes):
    vol_change = 0
    for pred_size in pred_sizes:
        vol_change += (np.pi/6)*(pred_size)**3
    area = im.shape[0]*nm_per_pixel*im.shape[1]*nm_per_pixel
    vol = area*image_thickness
    pred_swelling = 100*(vol_change / (vol - vol_change))
    return pred_swelling

def run_true_analysis(SAVE_PATH, MY_IMAGE, NM_PER_PIXEL, IMAGE_THICKNESS, NUM_CLASSES, CLASS_NAMES, anno, defect_metadata):
    true_classes, true_segmentations, true_boxes, im = get_true_data(anno=anno, 
                        image_name=MY_IMAGE, 
                        image_path=IMAGE_PATH,
                        save_path=SAVE_PATH, 
                        defect_metadata=defect_metadata,
                        visualize_image=True, 
                        save_images=True)
    
    true_density = get_density(im=im,
                               pred_classes=true_classes,
                               nm_per_pixel=NM_PER_PIXEL,
                               num_classes = NUM_CLASSES)
    true_sizes, true_shapes = get_sizes_and_shapes_image(pred_segmentations=true_segmentations,
                                                         pred_classes=true_classes,
                                                         nm_per_pixel=NM_PER_PIXEL,
                                                         num_classes = NUM_CLASSES)
    true_swelling = get_swelling(im=im,
                                 image_thickness=IMAGE_THICKNESS,
                                 nm_per_pixel=NM_PER_PIXEL,
                                 pred_sizes=true_sizes)
    if NUM_CLASSES == 1:
        num_preds = len(true_classes)
        avg_pred_size = round(np.mean(true_sizes), 3)
        std_pred_size = round(np.std(true_sizes), 3)
        avg_pred_shape = round(np.mean(true_shapes), 3)
        std_pred_shape = round(np.std(true_shapes), 3)
        pred_density = round(true_density, 6)
        pred_swelling = round(true_swelling, 2)
    else:
        num_preds = list()
        for i, cls in enumerate(CLASS_NAMES):
              num_preds.append(len(np.array(true_classes)[np.where(np.array(true_classes)==i)]))
        avg_true_size = [round(np.mean(sizes), 3) for sizes in true_sizes]
        std_true_size = [round(np.std(sizes), 3) for sizes in true_sizes]
        avg_true_shape = [round(np.mean(shapes), 3) for shapes in true_shapes]
        std_true_shape = [round(np.std(shapes), 3) for shapes in true_shapes]
        true_density = [round(dens, 6) for dens in true_density]

    data_dict = dict()
    if type(true_density) is float:
        true_density = [true_density]
    data_dict['true_classes'] = true_classes
    data_dict['true_boxes'] = true_boxes
    data_dict['true_segmentations'] = true_segmentations
    data_dict['true_sizes'] = true_sizes
    data_dict['true_shapes'] = true_shapes
    data_dict['true_density'] = true_density
    data_dict['true_swelling'] = [true_swelling]
    data_dict['num_true'] = [len(true_boxes)]
        
    df = pd.DataFrame().from_dict(data_dict, orient='index').T
    df.to_csv(os.path.join(SAVE_PATH, 'true_stats_'+MY_IMAGE[:-4]+'.csv'), index=False)

    return true_classes, true_boxes, true_segmentations, true_sizes, true_shapes, true_density, true_swelling

def run_pred_analysis(SAVE_PATH, MY_IMAGE, NM_PER_PIXEL, IMAGE_THICKNESS, NUM_CLASSES, CLASS_NAMES, im, outputs):
    pred_classes, pred_boxes, pred_segmentations, pred_scores = get_pred_data(instances=outputs['instances'])
    pred_density = get_density(im=im,
                               pred_classes=pred_classes,
                               nm_per_pixel=NM_PER_PIXEL,
                               num_classes = NUM_CLASSES)
    pred_sizes, pred_shapes = get_sizes_and_shapes_image(pred_segmentations=pred_segmentations,
                                                         pred_classes=pred_classes,
                                                         nm_per_pixel=NM_PER_PIXEL,
                                                         num_classes = NUM_CLASSES)
    pred_swelling = get_swelling(im=im,
                                 image_thickness=IMAGE_THICKNESS,
                                 nm_per_pixel=NM_PER_PIXEL,
                                 pred_sizes=pred_sizes)
    if NUM_CLASSES == 1:
        num_preds = len(pred_classes)
        avg_pred_size = round(np.mean(pred_sizes), 3)
        std_pred_size = round(np.std(pred_sizes), 3)
        avg_pred_shape = round(np.mean(pred_shapes), 3)
        std_pred_shape = round(np.std(pred_shapes), 3)
        pred_density = round(pred_density, 6)
        pred_swelling = round(pred_swelling, 2)
    else:
        num_preds = list()
        for i, cls in enumerate(CLASS_NAMES):
              num_preds.append(len(np.array(pred_classes)[np.where(np.array(pred_classes)==i)]))
        avg_pred_size = [round(np.mean(sizes), 3) for sizes in pred_sizes]
        std_pred_size = [round(np.std(sizes), 3) for sizes in pred_sizes]
        avg_pred_shape = [round(np.mean(shapes), 3) for shapes in pred_shapes]
        std_pred_shape = [round(np.std(shapes), 3) for shapes in pred_shapes]
        pred_density = [round(dens, 6) for dens in pred_density]

    data_dict = dict()
    if type(pred_density) is float:
        pred_density = [pred_density]
    data_dict['pred_classes'] = pred_classes
    data_dict['pred_boxes'] = pred_boxes
    data_dict['pred_segmentations'] = pred_segmentations
    data_dict['pred_scores'] = pred_scores
    data_dict['pred_sizes'] = pred_sizes
    data_dict['pred_shapes'] = pred_shapes
    data_dict['pred_density'] = pred_density
    data_dict['pred_swelling'] = [pred_swelling]
    data_dict['num_preds'] = [num_preds]
        
    df = pd.DataFrame().from_dict(data_dict, orient='index').T
    df.to_csv(os.path.join(SAVE_PATH, 'predicted_stats_'+MY_IMAGE[:-4]+'.csv'), index=False)


    return pred_classes, pred_boxes, pred_segmentations, pred_sizes, pred_shapes, pred_density, pred_swelling, pred_scores

def parse_box(box):
    if isinstance(box, str):
        # Remove brackets and split the string by comma
        box_list = box.strip('[]').split(',')
        # Convert each item to float and return
        return [float(x) for x in box_list]
    return box

def get_pred_f1(model_path, im, pred_boxes, pred_sizes, pred_scores):
    image_info_dict = {
        'area_ratio': [],
        'image_confidence': [],
        'avg obj conf': [],
        'avg obj size': [],
        'stdev obj conf': [],
        'stdev obj size': [],
        'counts': [],
    }
    for i in range(1, 10):
        image_info_dict[f'num_0.{i}'] = list()
    # Construct features
    bins = np.digitize(pred_scores, bins=np.arange(0.1, 1.0, 0.1))
    counts = np.bincount(bins, minlength=10)[1:]
    heights=np.zeros(len(pred_boxes))
    widths=np.zeros(len(pred_boxes))
    shape = (np.array(im)).shape[:2]
    b=0
    for box in pred_boxes:
        box=parse_box(box)
        x_min, y_min, x_max, y_max = np.asarray(box)
        heights[b]=x_max-x_min
        widths[b]=y_max-y_min
        b+=1
    for i, count in enumerate(counts, start=1):
        image_info_dict[f'num_0.{i}'].append(count)

    area_ratio=np.sum(heights*widths)/(shape[0]*shape[1])
    image_confidence=np.sum((heights*widths)*np.asarray(pred_scores))/np.sum(heights*widths)
    avg_size=np.mean(heights*widths)/(shape[0]*shape[1])
    std_size=np.std(heights*widths/(shape[0]*shape[1]))

    image_info_dict['area_ratio'].append(area_ratio)
    image_info_dict['counts'].append(len(pred_scores))
    image_info_dict['avg obj conf'].append(np.mean(pred_scores))
    image_info_dict['stdev obj conf'].append(np.std(pred_scores))
    image_info_dict['avg obj size'].append(avg_size)
    image_info_dict['stdev obj size'].append(std_size)
    image_info_dict['image_confidence'].append(image_confidence)

    X = np.array([image_info_dict['num_0.9'][0]/image_info_dict['counts'][0], # number fraction of detected defects with confidence score >0.9
             image_info_dict['counts'][0],  # total number of detected defects
             image_info_dict['area_ratio'][0],  # fraction of all detected defects' bounding boxes area over the image area
             image_info_dict['image_confidence'][0], # defect size weighted average confidence score
             image_info_dict['avg obj conf'][0],  # average confidence score
             image_info_dict['avg obj size'][0], #average defect bounding box size
             image_info_dict['stdev obj conf'][0], # standard deviation of confidence score
             image_info_dict['stdev obj size'][0]  # standard deviation of defect bounding box size
             ]).reshape(1, -1)

    #print(X.shape)

    import joblib
    scaler = joblib.load(os.path.join(model_path, 'StandardScaler.pkl'))
    X_scale = scaler.transform(X)

    rf = joblib.load(os.path.join(model_path, 'rfr_model_final.pkl'))
    #rf = joblib.load(os.path.join(model_path, 'RandomForestRegressor.pkl'))
    pred_f1 = rf.predict(X_scale)
    
    return pred_f1, image_confidence

def make_parity(SAVE_PATH, trues, preds, quantity='sizes'):
    if quanitity == 'sizes':
        units = 'nm'
    elif quantity == 'shapes':
        units = 'Heywood circularity'
    elif quantity == 'density':
        units = '#*10^4/nm^2'
    elif quantity == 'swelling':
        units = '% vol change'

    plt.clf()
    plt.scatter(trues, preds, s=80, color='blue', edgecolor='black', alpha=0.5)
    plt.xlabel('True average cavity '+quantity+' ('+units+')', fontsize=14)
    plt.xticks(fontsize=12)
    plt.ylabel('Pred average cavity '+quantity+' ('+units+')', fontsize=14)
    plt.yticks(fontsize=12)
    min1 = min(trues)
    min2 = min(preds)
    max1 = max(trues)
    max2 = max(preds)
    minn = min([min1, min2])
    maxx = max([max1, max2])
    plt.plot([minn, maxx], [minn, maxx], 'k--')

    r2 = r2_score(trues, preds)
    mae = mean_absolute_error(trues, preds)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    rmse_std = rmse / np.std(trues)

    plt.text(1.2*minn, 0.8*maxx, 'R2: '+str(round(r2, 3)), fontsize=10)
    plt.text(1.2*minn, 0.75*maxx, 'MAE: '+str(round(mae, 3)), fontsize=10)
    plt.text(1.2*minn, 0.7*maxx, 'RMSE: '+str(round(rmse, 3)), fontsize=10)
    plt.text(1.2*minn, 0.65*maxx, 'RMSE/std: '+str(round(rmse_std, 3)), fontsize=10)
    
    plt.savefig(os.path.join(SAVE_PATH, 'parity_true_pred_cavity_'+quantity+'.png'), dpi=300, bbox_inches='tight')

    df = pd.DataFrame({'True '+quantity: trues, 'Pred '+quantity: preds})
    df.to_csv(os.path.join(SAVE_PATH, 'parity_true_pred_cavity_'+quantity+'.csv'), index=False)        

    return

def run_predict(IMAGE_LIST, IMAGE_PATH, MODEL_PATH, MODEL_PATH_BASE, SAVE_PATH, NM_PER_PIXEL_LIST, IMAGE_THICKNESS_LIST, NUM_CLASSES, CLASS_NAMES, CLASS_COLORS, MAKE_SIZE_HIST=False):
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    predictor, defect_metadata = get_config(IMAGE_PATH, MODEL_PATH, SAVE_PATH, NUM_CLASSES, CLASS_NAMES, CLASS_COLORS)
    if len(NM_PER_PIXEL_LIST) == 1:
        if len(NM_PER_PIXEL_LIST) < len(IMAGE_LIST):
            print('You have specified multiple images but only one value for NM_PER_PIXEL_LIST. Applying this value to all images...')
            val = NM_PER_PIXEL_LIST[0]
            NM_PER_PIXEL_LIST = [val for _ in range(len(IMAGE_LIST))]
    if len(IMAGE_THICKNESS_LIST) == 1:
        if len(IMAGE_THICKNESS_LIST) < len(IMAGE_LIST):
            print('You have specified multiple images but only one value for IMAGE_THICKNESS_LIST. Applying this value to all images...')
            val = IMAGE_THICKNESS_LIST[0]
            IMAGE_THICKNESS_LIST = [val for _ in range(len(IMAGE_LIST))]
    for MY_IMAGE, NM_PER_PIXEL, IMAGE_THICKNESS in zip(IMAGE_LIST, NM_PER_PIXEL_LIST, IMAGE_THICKNESS_LIST):
        visualize_image(IMAGE_PATH, MY_IMAGE)
        im, outputs = visualize_pred_image(IMAGE_PATH, SAVE_PATH, MY_IMAGE, predictor, defect_metadata)
        pred_classes, pred_boxes, pred_segmentations, pred_sizes, pred_shapes, pred_density, pred_swelling, pred_scores = run_analysis(SAVE_PATH, MY_IMAGE, NM_PER_PIXEL, IMAGE_THICKNESS, NUM_CLASSES, CLASS_NAMES, im, outputs)

        pred_f1, image_confidence = get_pred_f1(MODEL_PATH_BASE, im, pred_boxes, pred_sizes, pred_scores)

        print('********** IMAGE PREDICTIONS **********')
        print(' Image name:', MY_IMAGE)
        print(' Defect types:', CLASS_NAMES)
        print(' Predicted F1 score (from random forest):', pred_f1[0])
        print(' Calculated image confidence score:', image_confidence)
        print(' Num predicted defects:', len(pred_boxes))
        print(' Pred swelling (percent swelling):', pred_swelling)
        print(' Pred defect density (#*10^4/nm^2):', pred_density[0])
        print(' Pred defect size (nm) as avg, stdev:', np.mean(pred_sizes), np.std(pred_sizes))
        print(' Pred defect shape (Heywood circularity) as avg, stdev:', np.mean(pred_shapes), np.std(pred_shapes))
        
        if MAKE_SIZE_HIST == True:
            plt.hist(pred_sizes, label=CLASS_NAMES)
            plt.xlabel('Defect size (nm)')
            plt.ylabel('Number of instances')
            plt.legend(loc='best')
            hist_name = 'size_histogram_'+MY_IMAGE[:-4]+'.png'
            plt.savefig(os.path.join(SAVE_PATH, hist_name), dpi=250, bbox_inches='tight')
            plt.show()
        
    return

def run_assess(ANNO_PATH, IMAGE_LIST, IMAGE_PATH, MODEL_PATH, MODEL_PATH_BASE, SAVE_PATH, NM_PER_PIXEL_LIST, IMAGE_THICKNESS_LIST, NUM_CLASSES, CLASS_NAMES, CLASS_COLORS, MAKE_SIZE_HIST=False)
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    predictor, defect_metadata = get_config(IMAGE_PATH, MODEL_PATH, SAVE_PATH, NUM_CLASSES, CLASS_NAMES, CLASS_COLORS)

    if len(NM_PER_PIXEL_LIST) == 1:
        if len(NM_PER_PIXEL_LIST) < len(IMAGE_LIST):
            print('You have specified multiple images but only one value for NM_PER_PIXEL_LIST. Applying this value to all images...')
            val = NM_PER_PIXEL_LIST[0]
            NM_PER_PIXEL_LIST = [val for _ in range(len(IMAGE_LIST))]
    if len(IMAGE_THICKNESS_LIST) == 1:
        if len(IMAGE_THICKNESS_LIST) < len(IMAGE_LIST):
            print('You have specified multiple images but only one value for IMAGE_THICKNESS_LIST. Applying this value to all images...')
            val = IMAGE_THICKNESS_LIST[0]
            IMAGE_THICKNESS_LIST = [val for _ in range(len(IMAGE_LIST))]

    with open(ANNO_PATH, 'rb') as f:
        anno = json.load(f)

    true_size_avg_img = list()
    true_density_avg_img = list()
    true_swelling_avg_img = list()
    true_shape_avg_img = list()
    pred_size_avg_img = list()
    pred_density_avg_img = list()
    pred_swelling_avg_img = list()
    pred_shape_avg_img = list()
    for MY_IMAGE, NM_PER_PIXEL, IMAGE_THICKNESS in zip(IMAGE_LIST, NM_PER_PIXEL_LIST, IMAGE_THICKNESS_LIST):
        # Get true annos
        true_classes, true_boxes, true_segmentations, true_sizes, true_shapes, true_density, true_swelling = run_true_analysis(SAVE_PATH, MY_IMAGE, NM_PER_PIXEL, IMAGE_THICKNESS, NUM_CLASSES, CLASS_NAMES, anno, defect_metadata)
        
        # Get preds
        im, outputs = visualize_pred_image(IMAGE_PATH, SAVE_PATH, MY_IMAGE, predictor, defect_metadata)
        pred_classes, pred_boxes, pred_segmentations, pred_sizes, pred_shapes, pred_density, pred_swelling, pred_scores = run_pred_analysis(SAVE_PATH, MY_IMAGE, NM_PER_PIXEL, IMAGE_THICKNESS, NUM_CLASSES, CLASS_NAMES, im, outputs)

        # Do detection to get found defects
        P = 'goat'
        R = 'sheep'
        F1 = 'cat'

        # Output report for the image
        print('********** IMAGE PREDICTIONS **********')
        print(' Image name:', MY_IMAGE)
        print(' Defect types:', CLASS_NAMES)
        print(' Num true defects:', len(true_boxes))
        print(' Num predicted defects:', len(pred_boxes))
        print(' Num found defects:', len(found_defects))
        print(' Image P, R, F1 scores:', P, R, F1)
        print(' True swelling (percent swelling):', true_swelling)
        print(' Pred swelling (percent swelling):', pred_swelling)
        print(' True defect density (#*10^4/nm^2):', true_density[0])
        print(' Pred defect density (#*10^4/nm^2):', pred_density[0])
        print(' True defect size (nm) as avg, stdev:', np.mean(true_sizes), np.std(true_sizes))
        print(' Pred defect size (nm) as avg, stdev:', np.mean(pred_sizes), np.std(pred_sizes))
        print(' True defect shape (Heywood circularity) as avg, stdev:', np.mean(true_shapes), np.std(true_shapes))
        print(' Pred defect shape (Heywood circularity) as avg, stdev:', np.mean(pred_shapes), np.std(pred_shapes))

        true_size_avg_img.append(np.mean(true_sizes))
        true_density_avg_img.append(true_density[0])
        true_swelling_avg_img.append(true_swelling)
        true_shape_avg_img.append(np.mean(true_shapes))
        pred_size_avg_img.append(np.mean(pred_sizes))
        pred_density_avg_img.append(pred_density[0])
        pred_swelling_avg_img.append(pred_swelling)
        pred_shape_avg_img.append(np.mean(pred_shapes))

    # Make parity plots with average values across all images
    make_parity(SAVE_PATH, true_sizes, pred_sizes, quantity='sizes')
        
    return
