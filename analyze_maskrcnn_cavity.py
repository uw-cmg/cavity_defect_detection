from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer, GenericMask
from detectron2.engine import DefaultPredictor

import pandas as pd
from pandas import ExcelWriter
import numpy as np
import cv2
import os
import itertools
import json
from pprint import pprint

from matplotlib.patches import Polygon
from matplotlib.figure import Figure, figaspect
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.font_manager import FontProperties

from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.stats import skew

from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score

from shapely.geometry import Polygon as shapelyPolygon

def save_xls(list_dfs, xls_path):
    with ExcelWriter(xls_path) as writer:
        for n, df in enumerate(list_dfs):
            df.to_excel(writer, 'defect_mask_' + str(n))
        writer.save()
    return

def get_mask_vertices(points):
    try:
        hull = ConvexHull(points)
    except:
        # Usually an error occurs if the input is too low-dimensional due to rounding of pixel values. Joggle input to help avoid this
        hull = ConvexHull(points, qhull_options='QJ')
    vertices = np.array([points[hull.vertices, 0], points[hull.vertices, 1]]).T
    return vertices

def get_fig_ax(aspect_ratio, x_align):
    w, h = figaspect(aspect_ratio)
    fig = Figure(figsize=(w, h))
    FigureCanvas(fig)

    # Set custom positioning, see this guide for more details:
    # https://python4astronomers.github.io/plotting/advanced.html
    left = 0.10
    bottom = 0.15
    right = 0.01
    top = 0.05
    width = x_align - left - right
    height = 1 - bottom - top
    ax = fig.add_axes((left, bottom, width, height), frameon=True)
    fig.set_tight_layout(False)
    return fig, ax

def plot_stats(fig, stats, x_align=0.65, y_align=0.90, font_dict=dict(), fontsize=10, type='float'):
    """
    Method that prints stats onto the plot. Goes off screen if they are too long or too many in number.

    Args:

        fig: (matplotlib figure object), a matplotlib figure object

        stats: (dict), dict of statistics to be included with a plot

        x_align: (float), float denoting x position of where to align display of stats on a plot

        y_align: (float), float denoting y position of where to align display of stats on a plot

        font_dict: (dict), dict of matplotlib font options to alter display of stats on plot

        fontsize: (int), the fontsize of stats to display on plot

    Returns:

        None

    """

    stat_str = '\n'.join(stat_to_string(name, value, nice_names=nice_names(), type=type)
                           for name,value in stats.items())

    fig.text(x_align, y_align, stat_str,
             verticalalignment='top', wrap=True, fontdict=font_dict, fontproperties=FontProperties(size=fontsize))

def stat_to_string(name, value, nice_names, type):
    """
    Method that converts a metric object into a string for displaying on a plot

    Args:

        name: (str), long name of a stat metric or quantity

        value: (float), value of the metric or quantity

    Return:

        (str), a string of the metric name, adjusted to look nicer for inclusion on a plot

    """

    " Stringifies the name value pair for display within a plot "
    if name in nice_names:
        name = nice_names[name]
    else:
        name = name.replace('_', ' ')

    # has a name only
    if not value:
        return name
    # has a mean and std
    if isinstance(value, tuple):
        mean, std = value
        if name == 'MAPE':
            return f'{name}:' + '\n\t' + f'{mean:3.2f}' + r'$\pm$' + f'{std:3.2f}'
        if name == 'R$^2$':
            return f'{name}:' + '\n\t' + f'{mean:3.2f}' + r'$\pm$' + f'{std:3.2f}'
        if name == 'Obs:Pred ratio':
            return f'{name}:' + '\n\t' + f'{mean:3.2f}' + r'$\pm$' + f'{std:3.2f}'
        else:
            return f'{name}:' + '\n\t' + f'{mean:3.2e}' + r'$\pm$' + f'{std:3.2e}'

    # has a name and value only
    if isinstance(value, int) or (isinstance(value, float) and value%1 == 0):
        return f'{name}: {int(value)}'
    if isinstance(value, float):
        if name == 'MAPE':
            return f'{name}: {value:3.2f}'
        if name == 'R$^2$':
            return f'{name}: {value:3.2f}'
        if name == 'Obs:Pred ratio':
            return f'{name}: {value:3.2f}'
        else:
            if type == 'float':
                return f'{name}: {value:3.2f}'
            elif type == 'scientific':
                return f'{name}: {value:3.2e}'
    return f'{name}: {value}' # probably a string

def nice_names():
    nice_names = {
    # classification:
    'accuracy': 'Accuracy',
    'f1_binary': '$F_1$',
    'f1_macro': 'f1_macro',
    'f1_micro': 'f1_micro',
    'f1_samples': 'f1_samples',
    'f1_weighted': 'f1_weighted',
    'log_loss': 'log_loss',
    'precision_binary': 'Precision',
    'precision_macro': 'prec_macro',
    'precision_micro': 'prec_micro',
    'precision_samples': 'prec_samples',
    'precision_weighted': 'prec_weighted',
    'recall_binary': 'Recall',
    'recall_macro': 'rcl_macro',
    'recall_micro': 'rcl_micro',
    'recall_samples': 'rcl_samples',
    'recall_weighted': 'rcl_weighted',
    'roc_auc': 'ROC_AUC',
    # regression:
    'explained_variance': 'expl_var',
    'mean_absolute_error': 'MAE',
    'mean_squared_error': 'MSE',
    'mean_squared_log_error': 'MSLE',
    'median_absolute_error': 'MedAE',
    'root_mean_squared_error': 'RMSE',
    'rmse_over_stdev': r'RMSE/$\sigma_y$',
    'R2': '$R^2$',
    'R2_noint': '$R^2_{noint}$',
    'R2_adjusted': '$R^2_{adjusted}$',
    'R2_fitted': '$R^2_{fitted}$'
    }
    return nice_names

def str2tuple(string):
    tup = tuple(map(int, string.split(', ')))
    return tup

def get_true_data_stats(cfg, defect_metadata, anno_dict_list_val, filename, model_checkpoint, true_and_pred_matching_threshold,
                        iou_score_threshold_test, show_images=False, save_images=False, save_all_data=False, mask_on=True):
    # Find the right image
    found_image = False
    for i, anno_dict in enumerate(anno_dict_list_val):
        base_filename = anno_dict["file_name"].split('/')[-1]
        if base_filename == filename:
            anno_dict = anno_dict_list_val[i]
            found_image = True
            print('Successfully found image', filename)
            break
    if found_image == False:
        print(
            'WARNING: An error occurred, the provided filename could not be corresponded with the name of an image file in the provided annotation dictionaries')
        return

    img = cv2.imread(anno_dict["file_name"])

    # Assign color to each defect in the image
    assigned_colors_list = list()
    for defect in anno_dict['annotations']:
        id = defect['category_id']
        if id == 0:  # void
            assigned_colors_list.append('r')
        else:
            print('WARNING: an error occurred. Should only have 1 defect type for cavity runs')
    anno_dict['assigned_colors'] = assigned_colors_list

    if mask_on == True:
        visualizer = Visualizer(img[:, :, ::-1],
                                metadata=defect_metadata,
                                scale=1.0,
                                instance_mode=ColorMode.SEGMENTATION
                                )

        vis = visualizer.draw_dataset_dict(anno_dict)
        img2 = vis.get_image()[:, :, ::-1]
        if show_images == True:
            cv2_imshow(img2)
        if save_images == True:
            cv2.imwrite(os.path.join(cfg.OUTPUT_DIR, filename + '_true_'+'TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_Checkpoint_'+str(model_checkpoint[:-4])+'.png'), img2)

    # PARSE AND OUTPUT TRUE PIXEL DATA
    df_list_true = list()
    true_classes = list()
    true_boxes = list()
    true_segmentations = list()
    n_masks = 0
    true_pixels_all = np.empty((anno_dict['height'], anno_dict['width']))
    true_pixels_all.fill(9999)
    for obj in anno_dict['annotations']:
        data_dict_true = dict()
        seg_y = list()
        seg_x = list()
        seg_y_nearestint = list()
        seg_x_nearestint = list()
        true_classes.append(obj['category_id'])
        true_boxes.append(obj['bbox'])

        n_defects = len(anno_dict['annotations'])
        if mask_on == False:
            for i, box in enumerate(true_boxes):
                # Note that there are no segmentations when mask is off. However, need to populate this list as it is
                # used later. It will carry through but not be used for any analysis.
                true_segmentations.append([[0, 0], [0, 0]])
        if mask_on == True:
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

            # Get pixels inside segmentation mask
            vertices = np.array(
                [[obj['bbox'][0], obj['bbox'][1]], [obj['bbox'][0], obj['bbox'][3]], [obj['bbox'][2], obj['bbox'][3]],
                 [obj['bbox'][2], obj['bbox'][1]]])
            # Make the path using the segmentation mask (polygon)
            poly = Polygon(segmentation)
            path = poly.get_path()
            x, y = np.meshgrid(np.arange(min(vertices[:, 0]) - 10, max(vertices[:, 0]) + 10),
                               np.arange(min(vertices[:, 1]) - 10, max(vertices[:, 1]) + 10))
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x, y)).T
            points_in_mask = path.contains_points(points)

            mask_list = list()
            defectid_list = list()
            pixels_y = list()
            pixels_x = list()
            for i, isin in enumerate(points_in_mask.tolist()):
                if isin == True:
                    pixels_y.append(points[i].tolist()[0])
                    pixels_x.append(points[i].tolist()[1])
            for n in range(len(pixels_y)):
                mask_list.append(n_masks)
                defectid_list.append(obj['category_id'])

            for y, x, defect in zip(pixels_y, pixels_x, defectid_list):
                if y < anno_dict['height']:
                    if x < anno_dict['width']:
                        true_pixels_all[int(y), int(x)] = defect

            # Put the true x, y pixels on the overall image array
            data_dict_true['mask'] = mask_list
            data_dict_true['defect ID'] = defectid_list
            data_dict_true['pixel_list_y'] = pixels_y
            data_dict_true['pixel_list_x'] = pixels_x
            df = pd.DataFrame.from_dict(data_dict_true, orient='index')
            df = df.transpose()
            df_list_true.append(df)
            n_masks += 1

    if save_all_data == True:
        if mask_on == True:
            print('Saving results to excel sheet...')
            save_xls(df_list_true, os.path.join(cfg.OUTPUT_DIR, filename + '_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_true_data_processed.xlsx'))

    print('Have true df list of size', len(true_boxes))

    return true_pixels_all, true_classes, true_segmentations, true_boxes


def get_pred_data_stats(cfg, defect_metadata, anno_dict_list_val, filename, predictor, model_checkpoint,
                        true_and_pred_matching_threshold, iou_score_threshold_test, show_images=False, save_images=False,
                        save_all_data=False, mask_on=True):
    #print('ANNO DICT VAL')
    #pprint(anno_dict_list_val)

    # Find the right image
    found_image = False
    for i, anno_dict in enumerate(anno_dict_list_val):
        base_filename = anno_dict["file_name"].split('/')[-1]
        if base_filename == filename:
            anno_dict = anno_dict_list_val[i]
            found_image = True
            print('Successfully found image', filename)
            break
    if found_image == False:
        print(
            'WARNING: An error occurred, the provided filename could not be corresponded with the name of an image file in the provided annotation dictionaries')
        return

    # Assign color to each defect in the image
    assigned_colors_list = list()
    for defect in anno_dict['annotations']:
        id = defect['category_id']
        if id == 0:  # void
            assigned_colors_list.append('r')
        else:
            print('WARNING: an error occurred. Should only have 1 defect type for cavity runs')
    anno_dict['assigned_colors'] = assigned_colors_list

    img = cv2.imread(anno_dict["file_name"])
    outputs = predictor(img)

    #print('ASSIGNED COLORS LENGTH')
    #print(len(assigned_colors_list))

    #print('OUTPUT INSTANCES')
    #pprint(outputs['instances'])

    #TODO: have visualizer also work to plot bounding boxes if mask_on = False.
    #if mask_on == True:
    visualizer = Visualizer(img[:, :, ::-1],
                            metadata=defect_metadata,
                            scale=1.0,
                            instance_mode=ColorMode.SEGMENTATION
                            )
    v = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    img3 = v.get_image()[:, :, ::-1]
    if show_images == True:
        cv2_imshow(img3)
    if save_images == True:
        cv2.imwrite(os.path.join(cfg.OUTPUT_DIR, filename + '_predicted_'+'TruePredMatch_'+str(true_and_pred_matching_threshold)+
                            '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_Checkpoint_'+str(model_checkpoint[:-4])+'.png'), img3)


    instances = outputs['instances']

    # PARSE AND OUTPUT PREDICTED PIXEL DATA
    # Organize predictions for each mask (i.e. predicted defect) and save as spreadsheet
    n_masks = 0
    df_list = list()
    pred_classes = list()
    pred_segmentations = list()
    pred_boxes = list()

    pred_pixels_all = np.empty((anno_dict['height'], anno_dict['width']))
    pred_pixels_all.fill(9999)

    if mask_on == False:
        pred_boxes = np.array(instances.pred_boxes).tolist()
        pred_boxes = [box.cpu().numpy().tolist() for box in pred_boxes]
        pred_classes = instances.pred_classes
        pred_classes = [int(c.cpu().numpy()) for c in pred_classes]
        pred_segmentations = [[[0, 0], [0, 0]] for c in pred_classes]

    if mask_on == True:
        # NEW WAY HERE 9/22/20
        #pred_boxes = np.array(instances.pred_boxes).tolist()
        #pred_boxes = [box.cpu().numpy().tolist() for box in pred_boxes]
        #pred_classes = instances.pred_classes
        #pred_classes = [int(c.cpu().numpy()) for c in pred_classes]
        ######

        #masks = np.asarray([mask.to("cpu") for mask in instances.pred_masks])
        #print('MASKS as array')
        #print(masks[0])
        #masks = [GenericMask(x, x.shape[0], x.shape[1]) for x in masks]
        #print('MASKS as GenericMask')
        #print(masks[0])
        #polygons = [m.polygons for m in masks]
        #print('POLYGONS')
        #print(polygons[0])

        for mask in instances.pred_masks:
            try:
                data_dict = dict()
                pred_coords_x = list()
                pred_coords_y = list()
                mask_id = list()
                defect_id = list()
                col_pixel = 0

                # NEW WAY HERE 9/22/20
                #print('MASK')
                #print(mask)
                #mask = np.array(mask.to('cpu'))
                #mask_generic = GenericMask(mask, mask.shape[0], mask.shape[1])
                #print('MASK as GenericMask')
                #print(mask_generic)
                #polygons = mask_generic.polygons
                #print('POLYGONS')
                #print(polygons[0])
                #pred_segmentations.append(polygons[0])
                #sys.exit()
                #############

                pred_boxes.append(np.array(instances.pred_boxes[n_masks].tensor.to("cpu")).tolist()[0])
                pred_classes.append(int(instances.pred_classes[n_masks].to("cpu")))
                #n_masks += 1

                pred_coords_x = np.where(mask.to("cpu"))[0].tolist()
                pred_coords_y = np.where(mask.to("cpu"))[1].tolist()

                for i in range(len(pred_coords_x)):
                    mask_id.append(n_masks)
                    defect_id.append(int(instances.pred_classes[n_masks].to("cpu")))

                # print('PRED COORDS SIZE VS IMAGE SIZE')
                # print(len(pred_coords_y), len(pred_coords_x), anno_dict['height'], anno_dict['width'])
                for y, x, defect in zip(pred_coords_y, pred_coords_x, defect_id):
                    if y < anno_dict['height']:
                        if x < anno_dict['width']:
                            pred_pixels_all[int(y), int(x)] = defect

                points = np.array([pred_coords_y, pred_coords_x])
                points = points.T
                vertices = get_mask_vertices(points)
                vertices_y = list(vertices[:, 0])
                vertices_x = list(vertices[:, 1])
                vertices_y, vertices_x = (list(t) for t in zip(*sorted(zip(vertices_y, vertices_x))))
                vertices = np.array([vertices_y, vertices_x]).T
                data_dict["y"] = pred_coords_y
                data_dict["x"] = pred_coords_x
                data_dict['segmentation_y'] = vertices[:, 0]
                data_dict['segmentation_x'] = vertices[:, 1]
                pred_segmentations.append([vertices[:, 0].tolist(), vertices[:, 1].tolist()])
                data_dict["mask"] = mask_id
                data_dict["defect ID"] = defect_id
                df = pd.DataFrame.from_dict(data_dict, orient='index')
                df = df.transpose()
                df_list.append(df)
                n_masks += 1
            except:
                print('FOUND ISSUE with IMAGE', filename)
                print('with bbox', np.array(instances.pred_boxes[n_masks].tensor.to("cpu")).tolist()[0])
                print('and object type', int(instances.pred_classes[n_masks].to("cpu")))
                n_masks += 1

    if save_all_data == True:
        if mask_on == True:
            print('Saving results to excel sheet...')
            save_xls(df_list, os.path.join(cfg.OUTPUT_DIR, filename + '_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_predicted_data_processed.xlsx'))

    print('Have pred df list of size', len(pred_boxes))

    return pred_pixels_all, pred_classes, pred_segmentations, pred_boxes

def get_predictor(cfg, model_checkpoint, iou_score_threshold_test, test_dataset_path):
    # Now, we perform inference with the trained model on the defect validation dataset. First, let's create a predictor using the model we just trained:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_checkpoint)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = iou_score_threshold_test  # set the testing threshold for this model
    cfg.DATASETS.TEST = (test_dataset_path,)
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    predictor = DefaultPredictor(cfg)
    return predictor

def get_pixel_classification_report(cfg, filename, true_pixels_all, pred_pixels_all):
    report = classification_report(true_pixels_all.flatten(), pred_pixels_all.flatten(), digits=3,
                                   target_names=['bdot', '111', '100', 'background'])
    report_asdict = classification_report(true_pixels_all.flatten(), pred_pixels_all.flatten(), digits=3,
                                          target_names=['bdot', '111', '100', 'background'], output_dict=True)
    print('PIXEL CLASSIFICATION REPORT')
    print(report)
    report_df = pd.DataFrame(report_asdict)
    report_df.to_excel(os.path.join(cfg.OUTPUT_DIR, filename + '_pixel_classification_report.xlsx'))
    return report_asdict

def get_class_classification_report(cfg, filename, true_classes, pred_classes):
    # cf = confusion_matrix(true_pixels_all.flatten(), pred_pixels_all.flatten())
    report = classification_report(true_classes, pred_classes, digits=3,
                                   target_names=['void'])
    report_asdict = classification_report(true_classes, pred_classes, digits=3,
                                          target_names=['void'], output_dict=True)
    print('CLASS CLASSIFICATION REPORT')
    print(report)
    report_df = pd.DataFrame(report_asdict)
    report_df.to_excel(os.path.join(cfg.OUTPUT_DIR, filename + '_class_classification_report.xlsx'))
    return report_asdict

def get_defect_size(segmentation, image_name, defect_type):
    
    images_lower_res = ['15', '16', '17', '18', '19']

    if image_name in images_lower_res:
        true_distance = 1 # 1 nm per pixel for these images
    elif image_name not in images_lower_res:
        true_distance = 0.38 # 0.38 nm per pixel for all other images
    else:
        print('Could not process image file name for density calculation')
        exit()

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

    # THIS IS NEW METHOD. Exp method: 111 and 100 loops just use the largest radius (because both are circular, just different orientation). Bdot is r = sqrt(r1*r2)
    if defect_type == 0:
        # This is void
        defect_radius_pixels = np.sqrt(min_distance*max_distance)
    else:
        print('WARNING: defect type other than 0 found. Should all be 1 type (cavities!)')
        print('Defect type', defect_type)
        exit()

    defect_radius_nm = defect_radius_pixels*true_distance
    defect_diameter_nm = 2*defect_radius_nm

    # Need to get defect shape factor. Using Heywood circularity factor, = perimeter / circumference of circle of same area
    # Need to use convex hull construction of segmentation mask, so points follow an ordered path, making the shape closed
    hull = ConvexHull(segmentation_xy, qhull_options='QJ')
    vertices = np.array([segmentation_xy[hull.vertices, 0], segmentation_xy[hull.vertices, 1]]).T
    polygon = shapelyPolygon(vertices)
    perimeter = polygon.length
    area = polygon.area
    radius = np.sqrt(area/np.pi)
    circumference = 2*np.pi*radius
    defect_shape_factor = perimeter/circumference
    if area == 0.0:
        print('FOUND AREA ZERO, HERE IS POLYGON and FULL SEG')
        print(polygon)
        print(segmentation_xy)

    return defect_diameter_nm, defect_shape_factor

def get_defect_number_densities(true_classes_all_flattened, pred_classes_all_flattened,
                                num_images, validation_image_filenames, model_checkpoint,
                                iou_score_threshold_test, true_and_pred_matching_threshold, save_path, save_to_file=False):

    images_lower_res = ['15', '16', '17', '18', '19']

    nm_per_pixel_large = 1
    nm_per_pixel_small = 0.38
    num_large = 0
    num_small = 0
    m_per_nm = 10**-9
    num_done = 0
    for image_name in validation_image_filenames:
        if num_done < num_images:
            print(image_name)
            if image_name in images_lower_res:
                num_large += 1  # 1 nm per pixel for these images
            elif image_name not in images_lower_res:
                num_small += 1  # 0.38 nm per pixel for all other images
            else:
                print('Could not process image file name for density calculation')
                exit()
            num_done += 1
    real_area = num_large*(nm_per_pixel_large*512)**2 + num_small*(nm_per_pixel_small*512)**2

    # Number of images determines total area to average over
    # Fill in correct value once you know it

    true_density = len(true_classes_all_flattened) / real_area
    pred_density = len(pred_classes_all_flattened) / real_area

    density_percenterror = 100 * abs(pred_density - true_density) / true_density

    datadict = {"true num": len(true_classes_all_flattened),
                "true density (#/nm^2)": true_density,
                "true density x 10^4 (#/nm^2)": true_density*10**4,
                "pred num": len(pred_classes_all_flattened),
                "pred density (#/nm^2)": pred_density,
                "pred density x 10^4 (#/nm^2)": pred_density*10**4,
                "percent error density": density_percenterror,
                "model checkpoint": model_checkpoint,
                "iou_score_threshold_test": iou_score_threshold_test,
                "true_and_pred_matching_threshold": true_and_pred_matching_threshold}
    # Save datadict to excel
    df_defectnumbers = pd.DataFrame().from_dict(datadict, orient='index')
    if save_to_file == True:
        df_defectnumbers.to_excel(save_path + '.xlsx')

    return df_defectnumbers

def get_overall_defect_stats(num_true_perimage, num_pred_perimage, num_found_perimage, model_checkpoint,
                                iou_score_threshold_test, true_and_pred_matching_threshold, save_path, save_to_file=False):
    # Total up the number of instances that are true, predicted, and found correctly for overall P, R, F1 scores
    num_true_total = np.sum(num_true_perimage)
    num_pred_total = np.sum(num_pred_perimage)
    num_found_total = np.sum(num_found_perimage)
    overall_fp = num_pred_total - num_found_total
    overall_fn = num_true_total - num_found_total
    overall_prec = num_found_total / (num_found_total + overall_fp)
    overall_recall = num_found_total / (num_found_total + overall_fn)
    overall_f1 = (2 * overall_prec * overall_recall) / (overall_prec + overall_recall)
    overall_stats_arr = np.array(
        [[num_true_total, num_pred_total, num_found_total, overall_prec, overall_recall, overall_f1, model_checkpoint, iou_score_threshold_test, true_and_pred_matching_threshold]])

    df_overallstats = pd.DataFrame(data=overall_stats_arr,
                                   columns=['num true total', 'num pred total', 'num found total',
                                            'overall precision', 'overall recall', 'overall F1', 'model_checkpoint',
                                            'iou_score_threshold_test', 'true_and_pred_matching_threshold'],
                                   index=['overall stats'])
    if save_to_file == True:
        df_overallstats.to_excel(save_path + '.xlsx')
    return df_overallstats

def get_defect_sizes_average_and_errors(true_defectsizes_nm, pred_defectsizes_nm, true_defectshapes, pred_defectshapes,
                                        model_checkpoint, iou_score_threshold_test, true_and_pred_matching_threshold,
                                        save_path, save_to_file, cfg, file_string):
    # Get average defect radius per defect, output to excel file
    average_true_defectsizes_nm = np.mean(true_defectsizes_nm)
    average_pred_defectsizes_nm = np.mean(pred_defectsizes_nm)

    average_true_defectshapes = np.mean(true_defectshapes)
    average_pred_defectshapes = np.mean(pred_defectshapes)

    percent_error_defectsizes = 100*abs(average_true_defectsizes_nm-average_pred_defectsizes_nm)/average_true_defectsizes_nm

    percent_error_defectshapes = 100*abs(average_true_defectshapes-average_pred_defectshapes)/average_true_defectshapes

    sizes_arr = np.array([[true_defectsizes_nm],
                          [len(true_defectsizes_nm)],
                          [average_true_defectsizes_nm],
                          [pred_defectsizes_nm],
                          [len(pred_defectsizes_nm)],
                          [average_pred_defectsizes_nm],
                          [percent_error_defectsizes],
                          [np.mean(percent_error_defectsizes)],
                          [percent_error_defectshapes],
                          [np.mean(percent_error_defectshapes)],
                          [model_checkpoint],
                          [iou_score_threshold_test],
                          [true_and_pred_matching_threshold]])

    df_defectsizes = pd.DataFrame(data=sizes_arr, columns=['cavity'],
                                  index=['true all diameter values',
                                         'number true diameter values',
                                         'true averaged diameter values',
                                         'predicted all diameter values',
                                         'number predicted diameter values',
                                         'predicted average diameter values',
                                         'percent error average diameter values',
                                         'average percent error all defect diameters',
                                         'percent error average shape values',
                                         'average percent error all defect shapes',
                                         'model_checkpoint',
                                         'iou_score_threshold_test',
                                         'true_and_pred_matching_threshold'])
    if save_to_file == True:
        df_defectsizes.to_excel(save_path + '.xlsx')

    #####
    #
    # Here, make histograms of true and predicted defect size distributions
    #
    #####
    # Cavity size histogram
    fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
    ax.hist(true_defectsizes_nm, bins=np.arange(0, 20, 2), color='b', edgecolor='k', alpha=0.50, label='True')
    ax.hist(pred_defectsizes_nm, bins=np.arange(0, 20, 2), color='g', edgecolor='k', alpha=0.50, label='Predicted')
    ax.set_xlabel('Cavity sizes (nm)', fontsize=12)
    ax.set_ylabel('Number of instances', fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(sorted(true_defectsizes_nm), range(len(true_defectsizes_nm)), 'b--', linewidth=1, label='True')
    ax2.set_ylabel('Total number of true cavities', fontsize=12)
    ax2.plot(sorted(pred_defectsizes_nm), range(len(pred_defectsizes_nm)), 'g--', linewidth=1, label='Predicted')
    ax2.set_ylabel('Total number of predicted cavities', fontsize=12)
    ax.legend(loc='lower right')
    true_skew = skew(true_defectsizes_nm)
    pred_skew = skew(pred_defectsizes_nm)
    true_stats = pd.DataFrame(true_defectsizes_nm).describe().to_dict()[0]
    true_stats['skew'] = true_skew
    pred_stats = pd.DataFrame(pred_defectsizes_nm).describe().to_dict()[0]
    pred_stats['skew'] = pred_skew
    plot_stats(fig, true_stats, x_align=0.725, y_align=0.90, type='float', font_dict={'color':'b'})
    plot_stats(fig, pred_stats, x_align=0.725, y_align=0.50, type='float', font_dict={'color':'g'})
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'AllImages_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_ObjectSizes_Cavity_'+str(file_string)+'.png'),
                dpi=250, bbox_inches='tight')

    #####
    #
    # Here, make histograms of true and predicted defect shape distributions
    #
    #####
    # Cavity shape histogram
    fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
    ax.hist(true_defectshapes, bins=np.arange(1, 2, 0.1), color='b', edgecolor='k', alpha=0.50, label='True')
    ax.hist(pred_defectshapes, bins=np.arange(1, 2, 0.1), color='g', edgecolor='k', alpha=0.50, label='Predicted')
    ax.set_xlabel('Cavity Heywood circularity', fontsize=12)
    ax.set_ylabel('Number of instances', fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(sorted(true_defectshapes), range(len(true_defectshapes)), 'b--', linewidth=1, label='True')
    ax2.set_ylabel('Total number of true cavities', fontsize=12)
    ax2.plot(sorted(pred_defectshapes), range(len(pred_defectshapes)), 'g--', linewidth=1, label='Predicted')
    ax2.set_ylabel('Total number of predicted cavities', fontsize=12)
    ax.legend(loc='lower right')
    true_skew = skew(true_defectshapes)
    pred_skew = skew(pred_defectshapes)
    true_stats = pd.DataFrame(true_defectshapes).describe().to_dict()[0]
    true_stats['skew'] = true_skew
    pred_stats = pd.DataFrame(pred_defectshapes).describe().to_dict()[0]
    pred_stats['skew'] = pred_skew
    plot_stats(fig, true_stats, x_align=0.725, y_align=0.90, type='float', font_dict={'color':'b'})
    plot_stats(fig, pred_stats, x_align=0.725, y_align=0.50, type='float', font_dict={'color':'g'})
    ax.set_yscale('log')
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'AllImages_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_ObjectShapes_Cavity_'+str(file_string)+'.png'),
                dpi=250, bbox_inches='tight')

    return df_defectsizes


def match_true_and_predicted_defects_iou_bbox(true_classes_all_oneimage_sorted, pred_classes_all_oneimage_sorted,
        true_segmentations_oneimage_sorted, pred_segmentations_oneimage_sorted,
        true_boxes_oneimage_sorted, pred_boxes_oneimage_sorted, num_found, true_defectsizes_nm_foundonly,
        pred_defectsizes_nm_foundonly, true_defectshapes_foundonly, pred_defectshapes_foundonly,
        image_name, mask_on=True, iou_threshold=0.5):

    # Loop over true bboxes and check if they correspond to pred bboxes. Do this by calculating IoU of all predicted boxes
    # and selecting the highest one. If not, then prediction missed one
    true_pred_index_list = list()
    for i, true_box in enumerate(true_boxes_oneimage_sorted):
        ious = dict()
        for j, pred_box in enumerate(pred_boxes_oneimage_sorted):
            if j not in true_pred_index_list:
                iou = bb_intersection_over_union(boxA=true_box, boxB=pred_box)
                #print('True box', true_box, 'and pred box', pred_box, 'have iou', iou)
                ious[j] = iou
        # Use whichever has largest iou
        iou = -1
        for k, v in ious.items():
            if v > iou:
                iou = v
                true_pred_index = k

        # Check that the iou satisfies the iou_threshold value set by user
        if iou >= iou_threshold:
            true_class = true_classes_all_oneimage_sorted[i]
            pred_class = pred_classes_all_oneimage_sorted[true_pred_index]

            # Found a defect, so overall a true positive (not discerning defect type)
            num_found += 1

            true_pred_index_list.append(true_pred_index)

            # Calculate the defect size since found a defect where there should be one
            if mask_on == True:
                true_defect_diameter, true_defect_shape_factor = get_defect_size(segmentation=true_segmentations_oneimage_sorted[i],
                                                       image_name=image_name, defect_type=true_class)
                pred_defect_diameter, pred_defect_shape_factor = get_defect_size(segmentation=pred_segmentations_oneimage_sorted[true_pred_index],
                                                       image_name=image_name, defect_type=pred_class)
            else:
                # Note that the use of the mask is required to get defect sizes
                true_defect_diameter = 0
                pred_defect_diameter = 0
                true_defect_shape_factor = 0
                pred_defect_shape_factor = 0

            true_defectsizes_nm_foundonly.append(true_defect_diameter)
            true_defectshapes_foundonly.append(true_defect_shape_factor)

            pred_defectsizes_nm_foundonly.append(pred_defect_diameter)
            pred_defectshapes_foundonly.append(pred_defect_shape_factor)

            print('FOUND TRUE BOX', [true_box[0], true_box[1], true_box[2], true_box[3]], 'with TRUE CLASS',
                  true_class, 'and PRED BOX',
                  [pred_boxes_oneimage_sorted[true_pred_index][0], pred_boxes_oneimage_sorted[true_pred_index][1],
                   pred_boxes_oneimage_sorted[true_pred_index][2], pred_boxes_oneimage_sorted[true_pred_index][3]], 'with PRED CLASS',
                  pred_class)
        else:
            print('FOUND TRUE BOX', [true_box[0], true_box[1], true_box[2], true_box[3]], 'with TRUE CLASS',
                  true_classes_all_oneimage_sorted[i], 'BUT NO CORRESPONDING PRED MASK THAT MET IOU THRESHOLD')

    return num_found, true_defectsizes_nm_foundonly, pred_defectsizes_nm_foundonly, \
            true_defectshapes_foundonly, pred_defectshapes_foundonly

def bb_intersection_over_union(boxA, boxB):
    # This code is not mine. I got it from this github post: https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc,
    # which corrected an issue in this original post: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0.0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def save_excel_together_finalreport(full_dict_dfs_per_IoUscorethreshold, sheet_names, save_path):
    """
    overall_stats_per_IoU (dict): keys are iou_score_threshold_test, values are lists of df's containing all analysis
    """
    with ExcelWriter(save_path) as writer:
        for iou_score_threshold_test, full_dict_dfs in full_dict_dfs_per_IoUscorethreshold.items():
            num_done = 0
            print('iou score thresh', iou_score_threshold_test)
            for model_checkpoint, dict_dfs in full_dict_dfs.items():
                for true_and_pred_matching_threshold, list_dfs in dict_dfs.items():
                    print('true/pred match tresh', true_and_pred_matching_threshold)
                    for sheet_name, df in zip(sheet_names, list_dfs):
                        print('on sheet name', sheet_name)
                        startrow_multiplier = df.shape[0]
                        if num_done == 0:
                            print(num_done, 0)
                            df.to_excel(writer, sheet_name+'_'+str(iou_score_threshold_test)+'_PredThresh', startrow=0)
                        else:
                            print(num_done, num_done*startrow_multiplier+2)
                            df.to_excel(writer, sheet_name+'_'+str(iou_score_threshold_test)+'_PredThresh', startrow=num_done*(startrow_multiplier+2))
                    num_done += 1
        writer.save()
    return

def save_excel_together_singlereport(list_dfs, sheet_names, save_path):
    with ExcelWriter(save_path) as writer:
        for sheet_name, df in zip(sheet_names, list_dfs):
             df.to_excel(writer, sheet_name)
        writer.save()
    return

def analysis_setup(test_dataset_path, input_yaml):
    for _, __, files in os.walk(test_dataset_path):
        validation_image_filenames = sorted(files)

    for _, __, files in os.walk(input_yaml['output_dir']):
        model_checkpoints_all = [f for f in files if '.pth' in f]
        if len(model_checkpoints_all) > 1:
            for model_checkpoint in model_checkpoints_all:
                if 'final' in model_checkpoint:
                    model_checkpoints_all.remove(model_checkpoint)

            # Get the subset of model checkpoints to analyze
            checkpoint_numbers_all = [int(ck.split('.pth')[0].split('_')[1]) for ck in model_checkpoints_all]
        else:
            checkpoint_numbers_all = [input_yaml['max_iter']]

    num_checkpoints = input_yaml['num_checkpoints']
    if input_yaml['num_checkpoints'] > len(model_checkpoints_all):
        print('Setting of num_checkpoints greater than number of model checkpoints, resetting to be all checkpoints in output path')
        num_checkpoints = len(model_checkpoints_all)

    if input_yaml['only_last_checkpoint'] == True:
        #checkpoint_numbers = [int(ck.split('.pth')[0].split('_')[1]) for ck in model_checkpoints]
        only_checkpoint_number = max(checkpoint_numbers_all)
        for model_checkpoint in model_checkpoints_all:
            if str(only_checkpoint_number) in model_checkpoint:
                only_checkpoint_name = model_checkpoint
        model_checkpoints = list()
        model_checkpoints.append(only_checkpoint_name)
        num_checkpoints = 1
        print('Only analyzing last checkpoint', only_checkpoint_name)
    else:
        model_checkpoints = list()
        for checkpoint_number in input_yaml['checkpoint_numbers']:
            for model_checkpoint in model_checkpoints_all:
                # TODO: have better logic here. Sometimes string matching will pick wrong checkpoint
                if str(checkpoint_number) in model_checkpoint:
                    model_checkpoints.append(model_checkpoint)

    num_images = input_yaml['num_images']
    if input_yaml['num_images'] > len(validation_image_filenames):
        print('Setting of num_images greater than number of validation images, resetting to be all validation images')
        num_images = len(validation_image_filenames)

    return validation_image_filenames, model_checkpoints, num_checkpoints, num_images

def plot_overall_stats_vs_iou_threshold(save_path, full_dict_dfs_per_IoUscorethreshold):
    # Need to parse the full_dict_dfs dictionary and get overall P, R, F1 as function of IoU pred threshold and true vs pred defect scoring
    IoUscorethresholds = list()
    TruevsPredthresholds = list()
    precisions_iouthresh = dict()
    recalls_iouthresh = dict()
    f1s_iouthresh = dict()
    precisions_truepredthresh = dict()
    recalls_truepredthresh = dict()
    f1s_truepredthresh = dict()
    for iou_score_threshold_test, full_dict_dfs in full_dict_dfs_per_IoUscorethreshold.items():
        if iou_score_threshold_test not in IoUscorethresholds:
            IoUscorethresholds.append(iou_score_threshold_test)
        for model_checkpoint, dict_dfs in full_dict_dfs.items():
            for true_and_pred_matching_threshold, list_dfs in dict_dfs.items():
                overall_stats = list_dfs[0]
                if iou_score_threshold_test not in precisions_iouthresh.keys():
                    precisions_iouthresh[iou_score_threshold_test]={true_and_pred_matching_threshold : float(np.array(overall_stats['overall precision'])[0])}
                    recalls_iouthresh[iou_score_threshold_test]={true_and_pred_matching_threshold: float(np.array(overall_stats['overall recall'])[0])}
                    f1s_iouthresh[iou_score_threshold_test]={true_and_pred_matching_threshold: float(np.array(overall_stats['overall F1'])[0])}
                else:
                    precisions_iouthresh[iou_score_threshold_test].update({true_and_pred_matching_threshold : float(np.array(overall_stats['overall precision'])[0])})
                    recalls_iouthresh[iou_score_threshold_test].update({true_and_pred_matching_threshold: float(np.array(overall_stats['overall recall'])[0])})
                    f1s_iouthresh[iou_score_threshold_test].update({true_and_pred_matching_threshold: float(np.array(overall_stats['overall F1'])[0])})
                if true_and_pred_matching_threshold not in precisions_truepredthresh.keys():
                    precisions_truepredthresh[true_and_pred_matching_threshold] = {iou_score_threshold_test: float(np.array(overall_stats['overall precision'])[0])}
                    recalls_truepredthresh[true_and_pred_matching_threshold] = {iou_score_threshold_test: float(np.array(overall_stats['overall recall'])[0])}
                    f1s_truepredthresh[true_and_pred_matching_threshold] = {iou_score_threshold_test: float(np.array(overall_stats['overall F1'])[0])}
                else:
                    precisions_truepredthresh[true_and_pred_matching_threshold].update({iou_score_threshold_test: float(np.array(overall_stats['overall precision'])[0])})
                    recalls_truepredthresh[true_and_pred_matching_threshold].update({iou_score_threshold_test: float(np.array(overall_stats['overall recall'])[0])})
                    f1s_truepredthresh[true_and_pred_matching_threshold].update({iou_score_threshold_test: float(np.array(overall_stats['overall F1'])[0])})
                if true_and_pred_matching_threshold not in TruevsPredthresholds:
                    TruevsPredthresholds.append(true_and_pred_matching_threshold)

    for iou_score_threshold_test in IoUscorethresholds:
        precisions_list = [v for v in precisions_iouthresh[iou_score_threshold_test].values()]
        recalls_list = [v for v in recalls_iouthresh[iou_score_threshold_test].values()]
        f1s_list = [v for v in f1s_iouthresh[iou_score_threshold_test].values()]
        fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)

        ax.plot(TruevsPredthresholds, precisions_list, color='red', marker='o', markersize= 12, linestyle='solid', linewidth=1.5, label='Precision')
        ax.plot(TruevsPredthresholds, recalls_list, color='green', marker='^', markersize= 12, linestyle='solid', linewidth=1.5, label='Recall')
        ax.plot(TruevsPredthresholds, f1s_list, color='blue', marker='H', markersize= 12, linestyle='solid', linewidth=1.5, label='F1')

        ax.set_xlabel('IoU Test vs. Predicted box threshold', fontsize=16)
        ax.set_ylabel('Performance metric', fontsize=16)
        ax.set_ylim(bottom=0.0, top=1.0)
        for tick in ax.xaxis.get_majorticklabels():
            tick.set_fontsize(14)
        for tick in ax.yaxis.get_majorticklabels():
            tick.set_fontsize(14)
        ax.legend(loc='best', fontsize=14, frameon=True)
        fig.savefig(os.path.join(save_path, 'Overall_Stats_vs_IoU_TestvsPredBox_threshold_'+str(iou_score_threshold_test)+'_IoUPredThreshold'+'.png'), dpi=200, bbox_inches='tight')

    for true_and_pred_matching_threshold in TruevsPredthresholds:
        precisions_list = [v for v in precisions_truepredthresh[true_and_pred_matching_threshold].values()]
        recalls_list = [v for v in recalls_truepredthresh[true_and_pred_matching_threshold].values()]
        f1s_list = [v for v in f1s_truepredthresh[true_and_pred_matching_threshold].values()]
        fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)

        ax.plot(IoUscorethresholds, precisions_list, color='red', marker='o', markersize= 12, linestyle='solid', linewidth=1.5, label='Precision')
        ax.plot(IoUscorethresholds, recalls_list, color='green', marker='^', markersize= 12, linestyle='solid', linewidth=1.5, label='Recall')
        ax.plot(IoUscorethresholds, f1s_list, color='blue', marker='H', markersize= 12, linestyle='solid', linewidth=1.5, label='F1')

        ax.set_xlabel('IoU predictor test threshold', fontsize=16)
        ax.set_ylabel('Performance metric', fontsize=16)
        ax.set_ylim(bottom=0.0, top=1.0)
        for tick in ax.xaxis.get_majorticklabels():
            tick.set_fontsize(14)
        for tick in ax.yaxis.get_majorticklabels():
            tick.set_fontsize(14)
        ax.legend(loc='best', fontsize=14, frameon=True)
        fig.savefig(os.path.join(save_path, 'Overall_Stats_vs_IoU_predictor_threshold_'+str(true_and_pred_matching_threshold)+'_TruevsPredBoxThreshold'+'.png'), dpi=200, bbox_inches='tight')

    return

def save_objecttotals_per_image(save_path, image_name, true_classes, pred_classes,
                                iou_score_threshold_test, true_and_pred_matching_threshold):
    # Make dict and save file of true and pred object totals
    total_true_objects = len(true_classes)
    total_pred_objects = len(pred_classes)
    data_dict = {'total_true_objects': total_true_objects,
                 'total_pred_objects': total_pred_objects}
    df = pd.DataFrame.from_dict(data=data_dict, orient='index')
    df.to_excel(os.path.join(save_path, image_name+'_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_ObjectTotals.xlsx'))
    return total_true_objects, total_pred_objects

def save_foundobjecttotals_per_image(save_path, image_name, num_found, total_true_objects, total_pred_objects,
                                     iou_score_threshold_test, true_and_pred_matching_threshold):
    # Make dict and save file of total found objects in image, regardless if defect type is correct
    fp = total_pred_objects - num_found
    fn = total_true_objects - num_found
    prec = num_found / (num_found + fp)
    recall = num_found / (num_found + fn)
    try:
        f1 = (2 * prec * recall) / (prec + recall)
    except:
        f1 = np.nan
    data_dict = {'total_true_objects': total_true_objects,
                 'total_pred_objects': total_pred_objects,
                 'total_found_objects': num_found,
                 'precision': prec,
                 'recall': recall,
                 'f1': f1}
    df = pd.DataFrame.from_dict(data=data_dict, orient='index')
    df.to_excel(os.path.join(save_path, image_name+'_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_FoundObjectTotals.xlsx'))
    return f1

def save_objectdensities_per_image(save_path, image_name, true_classes, pred_classes, iou_score_threshold_test,
                                   true_and_pred_matching_threshold):
    # Make dict and save file of true and pred object totals
    true_count = len(true_classes)
    pred_count = len(pred_classes)

    images_lower_res = ['15', '16', '17', '18', '19']

    nm_per_pixel_large = 1
    nm_per_pixel_small = 0.38
    num_large = 0
    num_small = 0

    if image_name in images_lower_res:
        print('FOUND LOW MAG IMAGE', image_name)
        num_large += 1  # 1 nm per pixel for these images
    elif image_name not in images_lower_res:
        print('FOUND HIGH MAG IMAGE', image_name)
        num_small += 1  # 0.38 nm per pixel for all other images
    else:
        print('Could not process image file name for density calculation')
        exit()
    real_area = num_large*(nm_per_pixel_large*512)**2 + num_small*(nm_per_pixel_small*512)**2
    print('HAVE REAL AREA', real_area, 'FOR IMAGE', image_name)

    true_density = true_count / real_area
    pred_density = pred_count / real_area
    percent_density_error = 100*(abs(np.mean(true_density)-np.mean(pred_density))/np.mean(true_density))
    data_dict = {'image area (nm^2)': real_area, 'true_count': true_count, 'true_density': true_density,
                 'pred_count': pred_count, 'pred_density': pred_density,
                 'percent_density_error': percent_density_error}
    df = pd.DataFrame.from_dict(data=data_dict, orient='index').T
    df.to_excel(os.path.join(save_path, image_name+'_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_ObjectDensities.xlsx'))
    return percent_density_error, true_density, pred_density

def save_objectsizes_per_image(save_path, image_name, true_classes, pred_classes, true_segs, pred_segs,
                               iou_score_threshold_test, true_and_pred_matching_threshold, mask_on):
    true_sizes = list()
    pred_sizes = list()
    true_shapes = list()
    pred_shapes = list()
    for seg, defect in zip(true_segs, true_classes):
        if mask_on == True:
            defect_size, defect_shape_factor = get_defect_size(segmentation=seg,
                                      image_name=image_name,
                                      defect_type=defect)
        else:
            defect_size = 0
            defect_shape_factor = 0

        true_sizes.append(defect_size)
        true_shapes.append(defect_shape_factor)

    for seg, defect in zip(pred_segs, pred_classes):
        if mask_on == True:
            defect_size, defect_shape_factor = get_defect_size(segmentation=seg,
                                      image_name=image_name,
                                      defect_type=defect)
        else:
            defect_size = 0
            defect_shape_factor = 0

        pred_sizes.append(defect_size)
        pred_shapes.append(defect_shape_factor)

    percent_size_error = 100*(abs(np.mean(true_sizes)-np.mean(pred_sizes))/np.mean(true_sizes))
    data_dict = {"true_sizes": true_sizes,
                 "pred_sizes": pred_sizes,
                 "true_avg_size": [np.mean(true_sizes)],
                 "true_stdev_size": [np.std(true_sizes)],
                 "pred_avg_size": [np.mean(pred_sizes)],
                 "pred_stdev_size": [np.std(pred_sizes)],
                 "true_shapes": true_shapes,
                 "pred_shapes": pred_shapes,
                 "true_avg_shape": [np.mean(true_shapes)],
                 "true_stdev_shape": [np.std(true_shapes)],
                 "pred_avg_shape": [np.mean(pred_shapes)],
                 "pred_stdev_shape": [np.std(pred_shapes)],
                 "percent_size_error": [percent_size_error]}
    df = pd.DataFrame.from_dict(data=data_dict, orient='index')
    df.to_excel(os.path.join(save_path, image_name+'_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_ObjectSizes.xlsx'))
    #####
    #
    # Make histograms of distribution of true and pred sizes for each defect type
    #
    #####
    # Histogram of sizes
    fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
    ax.hist(true_sizes, bins=np.arange(0, 20, 2), color='b', edgecolor='k', alpha=0.50, label='True')
    ax.hist(pred_sizes, bins=np.arange(0, 20, 2), color='g', edgecolor='k', alpha=0.50, label='Predicted')
    ax.set_xlabel('Cavity sizes (nm)', fontsize=12)
    ax.set_ylabel('Number of instances', fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(sorted(true_sizes), range(len(true_sizes)), 'b--', linewidth=1)
    ax2.set_ylabel('Total number of true cavities', fontsize=12)
    ax2.plot(sorted(pred_sizes), range(len(pred_sizes)), 'g--', linewidth=1)
    ax2.set_ylabel('Total number of predicted cavitities', fontsize=12)
    ax.legend(loc='lower right')
    try:
        true_skew = skew(true_sizes)
        pred_skew = skew(pred_sizes)
        true_stats = pd.DataFrame(true_sizes).describe().to_dict()[0]
        true_stats['skew'] = true_skew
        pred_stats = pd.DataFrame(pred_sizes).describe().to_dict()[0]
        pred_stats['skew'] = pred_skew
        plot_stats(fig, true_stats, x_align=0.725, y_align=0.90, type='float', font_dict={'color': 'b'})
        plot_stats(fig, pred_stats, x_align=0.725, y_align=0.50, type='float', font_dict={'color': 'g'})
    except:
        pass
    fig.savefig(os.path.join(save_path, image_name+'_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_ObjectSizes_Cavities'+'.png'),
                dpi=250, bbox_inches='tight')

    #####
    #
    # Make histograms of distribution of true and pred shapes for each defect type
    #
    #####
    # Histogram of cavity shapes
    fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
    ax.hist(true_shapes, bins=np.arange(1, 2, 0.1), color='b', edgecolor='k', alpha=0.50, label='True')
    ax.hist(pred_shapes, bins=np.arange(1, 2, 0.1), color='g', edgecolor='k', alpha=0.50, label='Predicted')
    ax.set_xlabel('Cavity Heywood circularity', fontsize=12)
    ax.set_ylabel('Number of instances', fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(sorted(true_shapes), range(len(true_shapes)), 'b--', linewidth=1)
    ax2.set_ylabel('Total number of true cavities', fontsize=12)
    ax2.plot(sorted(pred_shapes), range(len(pred_shapes)), 'g--', linewidth=1)
    ax2.set_ylabel('Total number of predicted cavitities', fontsize=12)
    ax.legend(loc='lower right')
    try:
        true_skew = skew(true_shapes)
        pred_skew = skew(pred_shapes)
        true_stats = pd.DataFrame(true_shapes).describe().to_dict()[0]
        true_stats['skew'] = true_skew
        pred_stats = pd.DataFrame(pred_shapes).describe().to_dict()[0]
        pred_stats['skew'] = pred_skew
        plot_stats(fig, true_stats, x_align=0.725, y_align=0.90, type='float', font_dict={'color': 'b'})
        plot_stats(fig, pred_stats, x_align=0.725, y_align=0.50, type='float', font_dict={'color': 'g'})
    except:
        pass
    fig.savefig(os.path.join(save_path, image_name+'_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_ObjectShapes_Cavity'+'.png'),
                dpi=250, bbox_inches='tight')

    return percent_size_error, np.mean(true_sizes), np.mean(pred_sizes), np.mean(true_shapes), np.mean(pred_shapes),\
           true_sizes, pred_sizes, true_shapes, pred_shapes

def analyze_checkpoints(cfg, defect_metadata, input_yaml, iou_score_threshold_test, test_dataset_path,
                        anno_dict_list_val, file_note, only_last_checkpoint=True, true_and_pred_matching_method='iou_bbox',
                        true_and_pred_matching_thresholds=[0.1]):

    '''
    true_and_pred_matching_method (str): must be one of "iou_bbox" and "pixelnorm_mask"
    true_and_pred_matching_thresholds (list): values of thresholds used to determine if true and predicted masks/bboxes
        correspond to the same defect. Values represent bbox IoUs and must be <=1 and >=0 if true_and_pred_matching_method=='iou_bbox',
        or values represent the norm of a pixel distance (typically values may be from 30 to 80) if true_and_pred_matching_method=='pixelnorm_mask'
    '''

    # Whether to save all pixel-wise data to excel sheets for each image
    save_all_data = False
    save_images = True

    validation_image_filenames, model_checkpoints, num_checkpoints, num_images = analysis_setup(test_dataset_path=test_dataset_path,
                                                                                                input_yaml=input_yaml)

    classification_reports_all_checkpoints_pixels = dict()
    dict_dfs = dict()
    full_dict_dfs = dict()
    checkpoints_done = 0
    for model_checkpoint in model_checkpoints:
        print('ANALYZING MODEL CHECKPOINT ', model_checkpoint)

        if checkpoints_done < num_checkpoints:
            # Here- loop over values of threshold to discern true and predicted masks/bboxes
            for true_and_pred_matching_threshold in true_and_pred_matching_thresholds:

                images_done = 0
                true_pixels_all = list()
                pred_pixels_all = list()
                true_classes_all = list()
                pred_classes_all = list()

                # Values needed to get overall TP, FP, FN values
                num_true_perimage = list()
                num_pred_perimage = list()
                num_found_perimage = list()

                true_defectsizes_nm_foundonly = list()
                pred_defectsizes_nm_foundonly = list()
                true_defectshapes_foundonly = list()
                pred_defectshapes_foundonly = list()

                true_defectsizes_nm_all = list()
                pred_defectsizes_nm_all = list()
                true_defectshapes_all = list()
                pred_defectshapes_all = list()

                percent_density_error_perimage_list = list()
                percent_size_error_perimage_list = list()
                true_density_perimage_list = list()
                pred_density_perimage_list = list()
                true_avg_size_perimage_list = list()
                pred_avg_size_perimage_list = list()
                true_avg_shape_perimage_list = list()
                pred_avg_shape_perimage_list = list()
                image_name_list = list()

                data_dict_per_image = dict()

                for filename in validation_image_filenames:
                    if images_done < num_images:
                        if filename not in list(data_dict_per_image.keys()):
                            data_dict_per_image[filename] = dict()

                        num_found = 0

                        predictor = get_predictor(cfg=cfg, model_checkpoint=model_checkpoint,
                                                  iou_score_threshold_test=iou_score_threshold_test,
                                                  test_dataset_path=test_dataset_path)

                        true_pixels_all_oneimage, true_classes_all_oneimage, true_segmentations_oneimage, \
                        true_boxes_oneimage = get_true_data_stats(cfg=cfg, defect_metadata=defect_metadata,
                                                                anno_dict_list_val=anno_dict_list_val,
                                                                filename=filename, model_checkpoint=model_checkpoint,
                                                                true_and_pred_matching_threshold=true_and_pred_matching_threshold,
                                                                iou_score_threshold_test=iou_score_threshold_test, show_images=False,
                                                                save_images=save_images, save_all_data=save_all_data,
                                                                mask_on=input_yaml['mask_on'])

                        #print('HAVE PREDICTOR')
                        #print(predictor)

                        pred_pixels_all_oneimage, pred_classes_all_oneimage, pred_segmentations_oneimage, \
                        pred_boxes_oneimage = get_pred_data_stats(cfg=cfg, defect_metadata=defect_metadata,
                                                                anno_dict_list_val=anno_dict_list_val,
                                                                filename=filename, predictor=predictor,
                                                                model_checkpoint=model_checkpoint,
                                                                true_and_pred_matching_threshold=true_and_pred_matching_threshold,
                                                                iou_score_threshold_test=iou_score_threshold_test,
                                                                show_images=False, save_images=save_images,
                                                                save_all_data=save_all_data,
                                                                mask_on=input_yaml['mask_on'])

                        print('SHAPES OF PRED BOXES, CLASSES, SEGS')
                        print(len(pred_boxes_oneimage), len(pred_classes_all_oneimage), len(pred_segmentations_oneimage))

                        true_segmentations_oneimage_sorted, true_classes_all_oneimage_sorted, true_boxes_oneimage_sorted = \
                            (list(t) for t in zip(*sorted(zip(true_segmentations_oneimage, true_classes_all_oneimage, true_boxes_oneimage))))
                        true_segmentations_oneimage_abbrev = list()
                        for i, true_seg in enumerate(true_segmentations_oneimage_sorted):
                            true_segmentations_oneimage_abbrev.append([true_seg[0][0], true_seg[1][0]])

                        pred_segmentations_oneimage_sorted, pred_classes_all_oneimage_sorted, pred_boxes_oneimage_sorted = \
                            (list(t) for t in zip(*sorted(zip(pred_segmentations_oneimage, pred_classes_all_oneimage, pred_boxes_oneimage))))
                        pred_segmentations_oneimage_abbrev = list()
                        for i, pred_seg in enumerate(pred_segmentations_oneimage_sorted):
                            pred_segmentations_oneimage_abbrev.append([pred_seg[0][0], pred_seg[1][0]])

                        # Here- using IoU of true and predicted bounding boxes to match true and predicted masks
                        if true_and_pred_matching_method == 'iou_bbox':
                            num_found, true_defectsizes_nm_foundonly, pred_defectsizes_nm_foundonly, \
                            true_defectshapes_foundonly, pred_defectshapes_foundonly = match_true_and_predicted_defects_iou_bbox(true_classes_all_oneimage_sorted, pred_classes_all_oneimage_sorted,
                                    true_segmentations_oneimage_sorted, pred_segmentations_oneimage_sorted,
                                    true_boxes_oneimage_sorted, pred_boxes_oneimage_sorted,
                                    num_found, true_defectsizes_nm_foundonly, pred_defectsizes_nm_foundonly,
                                    true_defectshapes_foundonly, pred_defectshapes_foundonly,
                                    image_name=filename, mask_on=input_yaml['mask_on'], iou_threshold =true_and_pred_matching_threshold)
                        else:
                            raise ValueError("true_and_pred_matching_method must be one of 'pixelnorm_mask' or 'iou_bbox'")

                        # Here, append number of defects found, true number of defects and predicted number of defects for this particular image
                        print('FOUND ', num_found, 'CORRECT INSTANCES FOUND')
                        num_found_perimage.append(num_found)
                        num_true_perimage.append(len(true_boxes_oneimage))
                        num_pred_perimage.append(len(pred_boxes_oneimage))

                        ########
                        #
                        # HERE save key per-image stats files here
                        #
                        ########
                        image_name = filename[:-4]
                        total_true_objects, total_pred_objects = save_objecttotals_per_image(save_path=cfg.OUTPUT_DIR,
                                                                                             image_name=image_name,
                                                                                             true_classes=true_classes_all_oneimage,
                                                                                             pred_classes=pred_classes_all_oneimage,
                                                                                             iou_score_threshold_test=iou_score_threshold_test,
                                                                                             true_and_pred_matching_threshold=true_and_pred_matching_threshold)

                        f1 = save_foundobjecttotals_per_image(save_path=cfg.OUTPUT_DIR,
                                                         image_name=image_name,
                                                         num_found=num_found,
                                                         total_true_objects=total_true_objects,
                                                         total_pred_objects=total_pred_objects,
                                                         iou_score_threshold_test=iou_score_threshold_test,
                                                         true_and_pred_matching_threshold=true_and_pred_matching_threshold)

                        data_dict_per_image[filename]['overall F1'] = f1

                        percent_density_error, true_density, pred_density = save_objectdensities_per_image(save_path=cfg.OUTPUT_DIR,
                                                       image_name=image_name,
                                                       true_classes=true_classes_all_oneimage,
                                                       pred_classes=pred_classes_all_oneimage,
                                                       iou_score_threshold_test=iou_score_threshold_test,
                                                       true_and_pred_matching_threshold=true_and_pred_matching_threshold
                                                       )
                        percent_density_error_perimage_list.append(percent_density_error)

                        true_density_perimage_list.append(true_density)
                        pred_density_perimage_list.append(pred_density)

                        data_dict_per_image[filename]['density error'] = percent_density_error
                        data_dict_per_image[filename]['avg density error'] = np.mean(percent_density_error)

                        percent_size_error, true_avg_size, pred_avg_size, true_avg_shape, pred_avg_shape, \
                        true_sizes, pred_sizes, true_shapes, pred_shapes = save_objectsizes_per_image(save_path=cfg.OUTPUT_DIR,
                                                   image_name=image_name,
                                                   true_classes=true_classes_all_oneimage,
                                                   pred_classes=pred_classes_all_oneimage,
                                                   true_segs=true_segmentations_oneimage,
                                                   pred_segs=pred_segmentations_oneimage,
                                                   iou_score_threshold_test=iou_score_threshold_test,
                                                   true_and_pred_matching_threshold=true_and_pred_matching_threshold,
                                                   mask_on=input_yaml['mask_on']
                                                   )
                        percent_size_error_perimage_list.append(percent_size_error)
                        true_avg_size_perimage_list.append(true_avg_size)
                        pred_avg_size_perimage_list.append(pred_avg_size)
                        true_avg_shape_perimage_list.append(true_avg_shape)
                        pred_avg_shape_perimage_list.append(pred_avg_shape)

                        data_dict_per_image[filename]['size error'] = percent_size_error
                        data_dict_per_image[filename]['avg size error'] = np.mean(percent_size_error)

                        true_defectsizes_nm_all += true_sizes
                        true_defectshapes_all += true_shapes
                        pred_defectsizes_nm_all += pred_sizes
                        pred_defectshapes_all += pred_shapes

                        true_pixels_all.append(true_pixels_all_oneimage.flatten().tolist())
                        pred_pixels_all.append(pred_pixels_all_oneimage.flatten().tolist())
                        true_classes_all.append(true_classes_all_oneimage)
                        pred_classes_all.append(pred_classes_all_oneimage)

                        image_name_list.append(filename)
                        images_done += 1

                # Make list of lists into one long list
                true_classes_all_flattened = list(itertools.chain(*true_classes_all))
                pred_classes_all_flattened = list(itertools.chain(*pred_classes_all))

                # Total up the number of instances that are true, predicted, and found correctly for overall P, R, F1 scores
                df_overallstats = get_overall_defect_stats(num_true_perimage=num_true_perimage,
                                                           num_pred_perimage=num_pred_perimage,
                                                           num_found_perimage=num_found_perimage,
                                                           model_checkpoint =model_checkpoint,
                                                           iou_score_threshold_test=iou_score_threshold_test,
                                                           true_and_pred_matching_threshold=true_and_pred_matching_threshold,
                                                           save_path=os.path.join(cfg.OUTPUT_DIR, 'OverallStats_' +
                                                            str(num_images) + '_Images' + str(model_checkpoint)[:-4] +
                                                            '_Checkpoint_' + str(iou_score_threshold_test) + '_IoUPredThreshold_'+
                                                                       str(true_and_pred_matching_threshold)+
                                                                       '_TruePredMatchThresh' +
                                                            '_RunType_' + str(file_note)),
                                                           save_to_file=False)


                df_defectnumbers = get_defect_number_densities(true_classes_all_flattened=true_classes_all_flattened,
                                                               pred_classes_all_flattened=pred_classes_all_flattened,
                                                               num_images=num_images,
                                                               validation_image_filenames=validation_image_filenames,
                                                               model_checkpoint=model_checkpoint,
                                                            iou_score_threshold_test=iou_score_threshold_test,
                                                               true_and_pred_matching_threshold=true_and_pred_matching_threshold,
                                                               save_path=os.path.join(cfg.OUTPUT_DIR,'DefectNumbers_' + str(
                                                                        num_images) + '_Images' + str(model_checkpoint)[:-4]
                                                                        + '_Checkpoint_' + str(iou_score_threshold_test) +
                                                                        '_IoUPredThreshold_' +
                                                                       str(true_and_pred_matching_threshold)+
                                                                       '_TruePredMatchThresh'+ '_RunType_' + str(file_note)),
                                                               save_to_file=False)

                df_defectsizes_FOUNDONLY = get_defect_sizes_average_and_errors(true_defectsizes_nm=true_defectsizes_nm_foundonly,
                                                                     pred_defectsizes_nm=pred_defectsizes_nm_foundonly,
                                                                    true_defectshapes=true_defectshapes_foundonly,
                                                                    pred_defectshapes=pred_defectshapes_foundonly,
                                                                     model_checkpoint=model_checkpoint,
                                                                     iou_score_threshold_test=iou_score_threshold_test,
                                                                     true_and_pred_matching_threshold=true_and_pred_matching_threshold,
                                                                    save_path=os.path.join(cfg.OUTPUT_DIR,
                                                                    'DefectSizes_FOUNDONLY' + str(num_images) + '_Images' + str(model_checkpoint)[:-4] +
                                                                    '_Checkpoint_' + str(iou_score_threshold_test) + '_IoUPredThreshold_' +
                                                                    str(true_and_pred_matching_threshold)+'_TruePredMatchThresh'+
                                                                    '_RunType_' + str(file_note)),
                                                                     save_to_file=False,
                                                                    cfg=cfg,
                                                                    file_string='FOUNDONLY')

                df_defectsizes_ALL = get_defect_sizes_average_and_errors(true_defectsizes_nm=true_defectsizes_nm_all,
                                                                     pred_defectsizes_nm=pred_defectsizes_nm_all,
                                                                         true_defectshapes=true_defectshapes_all,
                                                                         pred_defectshapes=pred_defectshapes_all,
                                                                     model_checkpoint=model_checkpoint,
                                                                     iou_score_threshold_test=iou_score_threshold_test,
                                                                     true_and_pred_matching_threshold=true_and_pred_matching_threshold,
                                                                    save_path=os.path.join(cfg.OUTPUT_DIR,
                                                                    'DefectSizes_ALL' + str(num_images) + '_Images' + str(model_checkpoint)[:-4] +
                                                                    '_Checkpoint_' + str(iou_score_threshold_test) + '_IoUPredThreshold_' +
                                                                    str(true_and_pred_matching_threshold)+'_TruePredMatchThresh'+
                                                                    '_RunType_' + str(file_note)),
                                                                     save_to_file=False,
                                                                    cfg=cfg,
                                                                    file_string='ALLDEFECTS')

                # Also make dataframes of defect size and density errors from per-image calculations
                data_dict_sizes = {'image names': image_name_list,
                            'cavity size percent error per image': percent_size_error_perimage_list,
                             'cavity average size percent error per image': [np.mean(percent_size_error_perimage_list)],
                             'cavity stdev size percent error per image': [np.std(percent_size_error_perimage_list)]
                             }
                df_defectsizes_perimage = pd.DataFrame.from_dict(data=data_dict_sizes, orient='index')

                data_dict_densities = {'image names': image_name_list,
                            'cavity density percent error per image': percent_density_error_perimage_list,
                             'cavity average density percent error per image': [np.mean(percent_density_error_perimage_list)],
                             'cavity stdev density percent error per image': [np.std(percent_density_error_perimage_list)]
                             }
                df_defectnumbers_perimage = pd.DataFrame.from_dict(data=data_dict_densities, orient='index')

                # At end of this checkpoint analysis, write final report Excel file with all relevant dfs
                list_dfs = [df_overallstats, df_defectsizes_FOUNDONLY, df_defectsizes_ALL, df_defectsizes_perimage,
                            df_defectnumbers, df_defectnumbers_perimage]

                dict_dfs[true_and_pred_matching_threshold] = list_dfs

                save_excel_together_singlereport(list_dfs=list_dfs,
                                                sheet_names=['OverallStats', 'DefectSizes_FoundOnly', 'DefectSizes_All',
                                                             'DefectSizes_PerImage',
                                                             'DefectNumbers', 'DefectNumbers_PerImage'],
                                                save_path=os.path.join(cfg.OUTPUT_DIR,
                                                                       'SingleReport_'+str(num_images)+'_Images_'+
                                                                       str(model_checkpoint)[:-4]+'_Checkpoint_'+str(iou_score_threshold_test)+
                                                                       '_IoUPredThreshold_'+
                                                                       str(true_and_pred_matching_threshold)+
                                                                       '_TruePredMatchThresh'+'_RunType_'+str(file_note)+'.xlsx'))

                ##########
                #
                # Here- make parity plots of true vs. pred avg defect sizes
                #
                ##########
                # cavity sizes
                fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
                true_avg_size_perimage_list = list(np.array(true_avg_size_perimage_list)[np.where(~np.isnan(np.array(pred_avg_size_perimage_list)))])
                pred_avg_size_perimage_list = list(np.array(pred_avg_size_perimage_list)[np.where(~np.isnan(np.array(pred_avg_size_perimage_list)))])
                true_avg_size_perimage_list = list(np.array(true_avg_size_perimage_list)[np.where(~np.isnan(np.array(true_avg_size_perimage_list)))])
                pred_avg_size_perimage_list = list(np.array(pred_avg_size_perimage_list)[np.where(~np.isnan(np.array(true_avg_size_perimage_list)))])
                ax.scatter(true_avg_size_perimage_list, pred_avg_size_perimage_list, color='blue', edgecolors='black', s=100, zorder=2, alpha=0.7)
                ax.set_xlabel('True avg cavity sizes per image (nm)', fontsize=12)
                ax.set_ylabel('Predicted avg cavity sizes per image (nm)', fontsize=12)
                xlow = int(min(true_avg_size_perimage_list) - 0.1*(max(true_avg_size_perimage_list)-min(true_avg_size_perimage_list)))
                xhigh = int(max(true_avg_size_perimage_list) + 0.1*(max(true_avg_size_perimage_list)-min(true_avg_size_perimage_list)))
                ax.set_xlim(left=xlow, right=xhigh)
                ax.set_ylim(bottom=xlow, top=xhigh)
                ax.plot([xlow, xhigh], [xlow, xhigh], color='black', lw=1.5, linestyle='--')
                r2 = r2_score(true_avg_size_perimage_list, pred_avg_size_perimage_list)
                mae = mean_absolute_error(true_avg_size_perimage_list, pred_avg_size_perimage_list)
                rmse = np.sqrt(mean_squared_error(true_avg_size_perimage_list, pred_avg_size_perimage_list))
                d = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
                plot_stats(fig, d, x_align=0.15, y_align=0.90, type='float')
                fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'DefectSize_ParityPlot_TruePredMatch_' + str(true_and_pred_matching_threshold) +
                                 '_IoUScoreThresh_' + str(iou_score_threshold_test) + '_Cavity' + '.png'), dpi=250, bbox_inches='tight')

                # All defect sizes avg and stdev parity plot
                true_sizes_avg = np.mean(true_avg_size_perimage_list)
                true_sizes_std = np.std(true_avg_size_perimage_list)
                pred_sizes_avg = np.mean(pred_avg_size_perimage_list)
                pred_sizes_std = np.std(pred_avg_size_perimage_list)
                fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
                ax.scatter(true_sizes_avg, pred_sizes_avg, color='blue',
                           edgecolors='black', s=100, zorder=2, alpha=0.7, label='cavity average')
                ax.errorbar(true_sizes_avg, pred_sizes_avg, xerr=true_sizes_std, yerr=pred_sizes_std, capsize=2, ecolor='k', linestyle='none', label=None)
                ax.legend(loc='lower right')
                ax.set_xlabel('True average cavity sizes (nm)', fontsize=12)
                ax.set_ylabel('Predicted average cavity sizes (nm)', fontsize=12)
                xlow = int(min(true_avg_size_perimage_list) - 0.1*(max(true_avg_size_perimage_list)-min(true_avg_size_perimage_list)))
                xhigh = int(max(true_avg_size_perimage_list) + 0.1*(max(true_avg_size_perimage_list)-min(true_avg_size_perimage_list)))
                ax.set_xlim(left=xlow, right=xhigh)
                ax.set_ylim(bottom=xlow, top=xhigh)
                ax.plot([xlow, xhigh], [xlow, xhigh], color='black', lw=1.5, linestyle='--')
                r2 = r2_score([true_sizes_avg], [pred_sizes_avg])
                mae = mean_absolute_error([true_sizes_avg], [pred_sizes_avg])
                rmse = np.sqrt(mean_squared_error([true_sizes_avg], [pred_sizes_avg]))
                d = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
                plot_stats(fig, d, x_align=0.15, y_align=0.90, type='float')
                fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'DefectSize_AvgStdev_ParityPlot_TruePredMatch_' + str(true_and_pred_matching_threshold) +
                                 '_IoUScoreThresh_' + str(iou_score_threshold_test) + '_AllDefects' + '.png'), dpi=250, bbox_inches='tight')


                ##########
                #
                # Here- make parity plots of true vs. pred  defect densities
                #
                ##########
                # cavity densities
                fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
                true_density_perimage_list = list(10**4*np.array(true_density_perimage_list)[np.where(~np.isnan(np.array(pred_density_perimage_list)) & ~np.isnan(np.array(true_density_perimage_list)))])
                pred_density_perimage_list = list(10**4*np.array(pred_density_perimage_list)[np.where(~np.isnan(np.array(pred_density_perimage_list)) & ~np.isnan(np.array(true_density_perimage_list)))])
                ax.scatter(true_density_perimage_list, pred_density_perimage_list, color='blue', edgecolors='black', s=100, zorder=2, alpha=0.7)
                ax.set_xlabel('True cavity densities per image (x10$^4$ #/nm$^2$)', fontsize=12)
                ax.set_ylabel('Predicted cavity densities per image (x10$^4$ #/nm$^2$)', fontsize=12)
                xlow = int(min(true_density_perimage_list) - 0.1*(max(true_density_perimage_list)-min(true_density_perimage_list)))
                xhigh = int(max(true_density_perimage_list) + 0.1*(max(true_density_perimage_list)-min(true_density_perimage_list)))
                ax.set_xlim(left=xlow, right=xhigh)
                ax.set_ylim(bottom=xlow, top=xhigh)
                ax.plot([xlow, xhigh], [xlow, xhigh], color='black', lw=1.5, linestyle='--')
                r2 = r2_score(true_density_perimage_list, pred_density_perimage_list)
                mae = mean_absolute_error(true_density_perimage_list, pred_density_perimage_list)
                rmse = np.sqrt(mean_squared_error(true_density_perimage_list, pred_density_perimage_list))
                d = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
                plot_stats(fig, d, x_align=0.15, y_align=0.90, type='float')
                fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'DefectDensity_ParityPlot_TruePredMatch_' + str(true_and_pred_matching_threshold) +
                                 '_IoUScoreThresh_' + str(iou_score_threshold_test) + '_Cavity' + '.png'), dpi=250, bbox_inches='tight')

                # All defect densities avg and stdev parity plot
                true_density_avg = np.mean(true_density_perimage_list)
                true_density_std = np.std(true_density_perimage_list)
                pred_density_avg = np.mean(pred_density_perimage_list)
                pred_density_std = np.std(pred_density_perimage_list)
                fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
                ax.scatter(true_density_avg, pred_density_avg, color='blue',
                           edgecolors='black', s=100, zorder=2, alpha=0.7, label='cavity average')
                ax.errorbar(true_density_avg, pred_density_avg, xerr=true_density_std, yerr=pred_density_std, capsize=2, ecolor='k', linestyle='none', label=None)
                ax.legend(loc='lower right')
                ax.set_xlabel('True average cavity densities (x10$^4$ #/nm$^2$)', fontsize=12)
                ax.set_ylabel('Predicted average cavity densities (x10$^4$ #/nm$^2$)', fontsize=12)
                xlow = int(min(true_density_perimage_list) - 0.1*(max(true_density_perimage_list)-min(true_density_perimage_list)))
                xhigh = int(max(true_density_perimage_list) + 0.1*(max(true_density_perimage_list)-min(true_density_perimage_list)))
                ax.set_xlim(left=xlow, right=xhigh)
                ax.set_ylim(bottom=xlow, top=xhigh)
                ax.plot([xlow, xhigh], [xlow, xhigh], color='black', lw=1.5, linestyle='--')
                r2 = r2_score([true_density_avg], [pred_density_avg])
                mae = mean_absolute_error([true_density_avg], [pred_density_avg])
                rmse = np.sqrt(mean_squared_error([true_density_avg], [pred_density_avg]))
                d = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
                plot_stats(fig, d, x_align=0.15, y_align=0.90, type='float')
                fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'DefectDensity_AvgStdev_ParityPlot_TruePredMatch_' + str(true_and_pred_matching_threshold) +
                                 '_IoUScoreThresh_' + str(iou_score_threshold_test) + '_AllDefects' + '.png'), dpi=250, bbox_inches='tight')

                ##########
                #
                # Here- make parity plots of true vs. pred avg defect shapes
                #
                ##########
                # cavity shapes
                fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
                true_avg_shape_perimage_list = list(np.array(true_avg_shape_perimage_list)[np.where(~np.isnan(np.array(pred_avg_shape_perimage_list)))])
                pred_avg_shape_perimage_list = list(np.array(pred_avg_shape_perimage_list)[np.where(~np.isnan(np.array(pred_avg_shape_perimage_list)))])
                true_avg_shape_perimage_list = list(np.array(true_avg_shape_perimage_list)[np.where(~np.isnan(np.array(true_avg_shape_perimage_list)))])
                pred_avg_shape_perimage_list = list(np.array(pred_avg_shape_perimage_list)[np.where(~np.isnan(np.array(true_avg_shape_perimage_list)))])
                ax.scatter(true_avg_shape_perimage_list, pred_avg_shape_perimage_list, color='blue', edgecolors='black', s=100, zorder=2, alpha=0.7)
                ax.set_xlabel('True avg cavity Heywood circularity per image', fontsize=12)
                ax.set_ylabel('Predicted avg cavity Heywood circularity per image', fontsize=12)
                xlow = 0.9
                xhigh = 1.1
                ax.set_xlim(left=xlow, right=xhigh)
                ax.set_ylim(bottom=xlow, top=xhigh)
                ax.plot([xlow, xhigh], [xlow, xhigh], color='black', lw=1.5, linestyle='--')
                r2 = r2_score(true_avg_shape_perimage_list, pred_avg_shape_perimage_list)
                mae = mean_absolute_error(true_avg_shape_perimage_list, pred_avg_shape_perimage_list)
                rmse = np.sqrt(mean_squared_error(true_avg_shape_perimage_list, pred_avg_shape_perimage_list))
                d = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
                plot_stats(fig, d, x_align=0.15, y_align=0.90, type='float')
                fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'DefectShape_ParityPlot_TruePredMatch_' + str(true_and_pred_matching_threshold) +
                                 '_IoUScoreThresh_' + str(iou_score_threshold_test) + '_Cavity' + '.png'), dpi=250, bbox_inches='tight')

                # All defect sizes avg and stdev parity plot
                true_shape_avg = np.mean(true_avg_shape_perimage_list)
                true_shape_std = np.std(true_avg_shape_perimage_list)
                pred_shape_avg = np.mean(pred_avg_shape_perimage_list)
                pred_shape_std = np.std(pred_avg_shape_perimage_list)
                fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
                ax.scatter(true_shape_avg, pred_shape_avg, color='blue',
                           edgecolors='black', s=100, zorder=2, alpha=0.7, label='cavity average')
                ax.errorbar(true_shape_avg, pred_shape_avg, xerr=true_shape_std, yerr=pred_shape_std, capsize=2, ecolor='k',
                            linestyle='none', label=None)
                ax.legend(loc='lower right')
                ax.set_xlabel('True average cavity Heywood circularity', fontsize=12)
                ax.set_ylabel('Predicted average cavity Heywood circularity', fontsize=12)
                xlow = 0.9
                xhigh = 1.1
                ax.set_xlim(left=xlow, right=xhigh)
                ax.set_ylim(bottom=xlow, top=xhigh)
                ax.plot([xlow, xhigh], [xlow, xhigh], color='black', lw=1.5, linestyle='--')
                r2 = r2_score([true_shape_avg], [pred_shape_avg])
                mae = mean_absolute_error([true_shape_avg], [pred_shape_avg])
                rmse = np.sqrt(mean_squared_error([true_shape_avg], [pred_shape_avg]))
                d = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
                plot_stats(fig, d, x_align=0.15, y_align=0.90, type='float')
                fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'DefectShape_AvgStdev_ParityPlot_TruePredMatch_' + str(
                    true_and_pred_matching_threshold) +'_IoUScoreThresh_' + str(iou_score_threshold_test) + '_AllDefects' + '.png'),
                            dpi=250, bbox_inches='tight')

                # Output all key per-image stats to json file
                with open(os.path.join(cfg.OUTPUT_DIR, 'StatsPerImage_TruePredMatch_' + str(true_and_pred_matching_threshold) +'_IoUScoreThresh_' + str(iou_score_threshold_test)+'.json'), 'w') as f:
                    json.dump(data_dict_per_image, f)

                # Get best and worst image for each statistic, output to file
                best_f1 = 0
                best_density = 10**5
                best_size = 10**5
                data_dict_best_images = dict()
                for img, data in data_dict_per_image.items():
                    if data['avg size error'] < best_size:
                        best_size = data['avg size error']
                        data_dict_best_images['best avg size error'] = img
                    if data['avg density error'] < best_density:
                        best_density = data['avg density error']
                        data_dict_best_images['best avg density error'] = img
                    if data['overall F1'] > best_f1:
                        best_f1 = data['overall F1']
                        data_dict_best_images['best overall F1'] = img

                with open(os.path.join(cfg.OUTPUT_DIR, 'BestImagesPerStat_TruePredMatch_' + str(true_and_pred_matching_threshold) +'_IoUScoreThresh_' + str(iou_score_threshold_test)+'.json'), 'w') as f:
                    json.dump(data_dict_best_images, f)

                worst_f1 = 1
                worst_density = 0
                worst_size = 0
                data_dict_worst_images = dict()
                for img, data in data_dict_per_image.items():
                    if data['avg size error'] > worst_size:
                        worst_size = data['avg size error']
                        data_dict_worst_images['worst avg size error'] = img
                    if data['avg density error'] > worst_density:
                        worst_density = data['avg density error']
                        data_dict_worst_images['worst avg density error'] = img
                    if data['overall F1'] < worst_f1:
                        worst_f1 = data['overall F1']
                        data_dict_worst_images['worst overall F1'] = img

                with open(os.path.join(cfg.OUTPUT_DIR, 'WorstImagesPerStat_TruePredMatch_' + str(true_and_pred_matching_threshold) + '_IoUScoreThresh_' + str(iou_score_threshold_test) + '.json'), 'w') as f:
                    json.dump(data_dict_worst_images, f)

                # Finished one true_and_pred_matching_threshold loop
        # Finished one checkpoint loop
        full_dict_dfs[model_checkpoint] = dict_dfs
        checkpoints_done += 1

    return full_dict_dfs, classification_reports_all_checkpoints_pixels

