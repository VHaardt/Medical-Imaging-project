# University project for "Medical Imaging and Big Data" course
---
Here you will find all the materials used to create the project. Below is a brief overview of the available notebooks and folders:

Python files:
- dataset: used to extract images, split data into train/test/validation, transform images 40% of the time, and add mask if required.
- unet_model_EF.py: parameters used for each layer for Early Fusion model.
- unet_model_MF_mask.py: parameters used for each layer for Middle Fusion model applying mask as a new band.
- unet_model_MF.py: parameters used for each layer for the Middle Fusion model.
- unet_parts.py: U-Net architecture.

Notebook:
- Early-Fusion_Mask.ipynb: contains the training of the Early Fusion model with the mask applied.
- Early-Fusion.ipynb: contains the training of the Early Fusion model.
- Middle-Fusion_Mask.ipynb: contains the training of the Middle Fusion model with the mask applied.
- Middle-Fusion.ipynb: contains the training of the Middle Fusion model.
- Slices_extraction.ipynb: Used to select 40% of the images (of a type), especially taking those with the presence of the tumor with at least 0.5% of pixels highlighted.
