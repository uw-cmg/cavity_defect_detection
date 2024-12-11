# Cavity defect detection in electron microscopy images
Mask R-CNN model to detect cavity defects in electron microscopy images. Also included is a random forest model to predict the detection F1 score as a measure of prediction performance

## Run example notebooks in Google Colab:

Make predictions of image F1 score with Random forest model and object predictions with Mask R-CNN using final model fit to all data.
This model was trained on all 770 images in our joint database.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uw-cmg/cavity_defect_detection/blob/main/Detectron_predict_v3_cavity.ipynb)

Assess predictions of object predictions with Mask R-CNN on held-out test set. This model was trained on 646 images and then assessed on a set of 124 held-out images
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uw-cmg/cavity_defect_detection/blob/main/Cavity_assess_model_predictions.ipynb)

## Citations

If you find this repository useful, please cite the following publications:

Li, N., Jacobs, R., Lynch, M., Agrawal, V., Field, K. G., Morgan, D., "Predicting Performance of Object Detection Models in Electron Microscopy using Random Forests", Submitted to Digital Discovery (2024)

Jacobs, R., Patki, P., Lynch, M., Chen, S., Morgan, D., Field, K. G. "Materials swelling revealed through automated semantic segmentation of cavities in electron microscopy images", Scientific Reports, 13, 5178 (2023). https://doi.org/10.1038/s41598-023-32454-2
