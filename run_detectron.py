import yaml
import argparse
import os
import shutil
from detectron_maskrcnn_cavity import train_detectron_maskrcnn, get_config_file, get_defect_metadata, get_defect_anno_dict_train, get_defect_anno_dict_val
from analyze_maskrcnn_cavity import analyze_checkpoints, plot_learning_curve, plot_overall_stats_vs_iou_threshold, save_excel_together_finalreport

def run_maskrcnn(input_yaml):
    cfg, defect_metadata = train_detectron_maskrcnn(input_yaml)
    #train_detectron_maskrcnn(input_yaml)
    return cfg, defect_metadata
    #return

def analyze_maskrcnn(cfg, defect_metadata, input_yaml, analyze_test=True, analyze_train=False):
    #cfg = get_config_file(input_yaml)
    #defect_metadata = get_defect_metadata()

    iou_score_threshold_tests = input_yaml['iou_score_threshold_test']

    if type(iou_score_threshold_tests) is float:
        iou_score_threshold_tests_list = list()
        iou_score_threshold_tests_list.append(iou_score_threshold_tests)
        iou_score_threshold_tests = iou_score_threshold_tests_list

    # Run over all IoU tests
    # Run analysis over all model checkpoints for each IoU test
    classification_reports_all_checkpoints_test_per_IoU = dict()
    classification_reports_all_checkpoints_train_per_IoU = dict()
    overall_stats_per_IoU = dict()
    full_dict_dfs_per_IoUscorethreshold = dict()
    for iou_score_threshold_test in iou_score_threshold_tests:

        if analyze_test == True:
            print('ANALYZING TEST')
            full_dict_dfs, classification_reports_all_checkpoints_pixels_test = \
                                                            analyze_checkpoints(cfg, defect_metadata, input_yaml,
                                                                 iou_score_threshold_test,
                                                                 test_dataset_path=input_yaml['test_dataset_path'],
                                                                 anno_dict_list_val=get_defect_anno_dict_val(test_annotations_path=input_yaml['test_annotations_path'],
                                                                                                             test_dataset_path=input_yaml['test_dataset_path']),
                                                                 file_note='Test', only_last_checkpoint=True, true_and_pred_matching_method='iou_bbox')
            # Use class data for per IoU plot
            #classification_reports_all_checkpoints_test_per_IoU[iou_score_threshold_test] = classification_reports_all_checkpoints_classes_test
            #overall_stats_per_IoU[iou_score_threshold_test] = list_dfs[0]
            full_dict_dfs_per_IoUscorethreshold[iou_score_threshold_test] = full_dict_dfs


        if analyze_train == True:
            print('ANALYZING TRAIN')
            # Switching to be training data set for training data classification reports
            cfg.DATASETS.TEST = (input_yaml['train_dataset_path'],)
            full_dict_dfs_train, classification_reports_all_checkpoints_pixels_train = analyze_checkpoints(cfg, defect_metadata, input_yaml,
                                                                 iou_score_threshold_test,
                                                                 test_dataset_path=input_yaml['train_dataset_path'],
                                                                 anno_dict_list_val=get_defect_anno_dict_train(train_annotations_path=input_yaml['train_annotations_path'],
                                                                                                             train_dataset_path=input_yaml['train_dataset_path']),
                                                                 file_note='Train', only_last_checkpoint=True, true_and_pred_matching_method='iou_bbox')
            # Use class data for per IoU plot
            #classification_reports_all_checkpoints_train_per_IoU[iou_score_threshold_test] = classification_reports_all_checkpoints_classes_train

        # Using classification reports from all checkpoints, plot learning curve of overall P, R, F1 vs model iteration
        if (analyze_test == True and analyze_train == True):
            plot_learning_curve(cfg, classification_reports_all_checkpoints_pixels_train, classification_reports_all_checkpoints_pixels_test)

    # Plot the overall defect stats P, R, F1 vs. iou_score_threshold_test value. This value alters how the predictions
    # are made and should be viewed as a hyperparameter. Typically, a value near 0.3-0.5 is likely best.

    plot_overall_stats_vs_iou_threshold(save_path=cfg.OUTPUT_DIR, full_dict_dfs_per_IoUscorethreshold=full_dict_dfs_per_IoUscorethreshold)

    # Catalog all of the results together over the IoU predictor thresholds into one spreadsheet
    save_excel_together_finalreport(full_dict_dfs_per_IoUscorethreshold=full_dict_dfs_per_IoUscorethreshold,
                                        #sheet_names=['OverallStats', 'DefectSizes', 'DefectSizes_PerImage',
                                        #             'DefectNumbers', 'DefectNumbers_PerImage',
                                        #'CM_DefectID', 'StatsDefectID'],
                                    sheet_names=['OverallStats', 'DefectSizes_Found', 'DefectSizes_All',
                                                 'Sizes_PerImage',
                                                 'DefectNumbers', 'Numbers_PerImage',
                                                 'CMDefectID', 'StatsDefectID'],
                                    save_path=os.path.join(cfg.OUTPUT_DIR, 'FinalReport_AllDefectDataAnalysis.xlsx'))

    return

def get_commandline_args():
    parser = argparse.ArgumentParser(description='Detectron Training')
    parser.add_argument('input', type=str, help='path to input file for Detectron training')
    args = parser.parse_args()
    return args.input

def move_files(input_yaml):
    cwd = os.getcwd()
    output_dir = input_yaml['output_dir']
    log = os.path.join(cwd, 'detectron_training_'+str(input_yaml['output_dir'])+'.log')
    yaml = os.path.join(cwd, str(input_yaml['output_dir'])+'.yaml')
    submit = os.path.join(cwd, 'submit_detectron_'+str(input_yaml['output_dir'])+'.sh')
    shutil.move(log, output_dir)
    shutil.move(yaml, output_dir)
    shutil.move(submit, output_dir)
    return

def get_input_yaml(input_file_path):
    with open(input_file_path, 'r') as f:
        input_yaml = yaml.load(f, Loader=yaml.FullLoader)
    return input_yaml

if __name__=='__main__':
    input = get_commandline_args()
    input_yaml = get_input_yaml(input_file_path=input)
    if input_yaml['run_maskrcnn'] == True:
        print('RUNNING MASKRCNN')
        cfg, defect_metadata = run_maskrcnn(input_yaml)
        print('defect metadata')
        print(defect_metadata)
        #run_maskrcnn(input_yaml)
    if input_yaml['analyze_maskrcnn'] == True:
        print('ANALYZING MASKRCNN')
        analyze_maskrcnn(cfg, defect_metadata, input_yaml, analyze_test=True, analyze_train=False)
        #analyze_maskrcnn(input_yaml)
    # Move input .yaml and final .log file to output dir
    #move_files(input_yaml)

