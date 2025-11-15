Sentinel-2Driven Land Use Classification
Using IRUNet,UNet,ResUNet for the
Perungudi Region



Accurate land use and land cover (LULC) classification is a critical baseline for management in sustainable urban planning, peri-urban monitoring, and environmental governance for urbanized landscapes that are rapidly changing. This paper proposes an integrated method to apply mapping and spatio-temporal analysis of LULC in the Perungudi area of Chennai and emphasizes temporal analysis as a way to understand urbanization and zones for amplifying urbanization. This pipeline involves a combination of manual annotation for ground truth preparation, conventional machine learning (Random Forest), and a high-performing deep ensemble framework (IRUNet - InceptionResNetV2 + UNet), engineered for addressing challenges presented by heterogenous urban landscape characteristics, seasonal change, and imbalancing classes of LULC outputs driven by atypical urbanization patterns. All spatial outputs (classified maps, statistics, and metrics) are systematically saved in a MongoDB database, which supports geo-tagged storage, and enables the extraction of data for interactive and visual FDA dashboards. The proposed workflow shows an improvement in segmentation accuracy and change-detection sensitivity compared to pixel-based protocols and provides an end-to-end approach for reproducible, scalable, and policy-relevant mapping of LULC to fast-growing peri-urban regions in India.

22MIA1171 MEDHAA M
22MIA1087     JANANI.P

Base paper reference



https://www.nature.com/articles/s41598-025-12512-7



Tools and libraries used
Python Libraries
Category Libraries Usage
Numerical / Array Operations	numpy, pandas	Data loading, preprocessing
Visualization	matplotlib, seaborn, random	Plotting sample images, masks, training history
Deep Learning Framework	tensorflow, keras	Building, training, evaluating segmentation models
Image Processing	cv2 (OpenCV)	Loading images, resizing, augmentations
Dataset Handling	os, glob	Reading dataset folders
Metrics	sklearn.metrics	Accuracy, F1-score, confusion matrix (if used)
Augmentation	Custom functions or imgaug / built-in Keras preprocessing	Flip, rotate, scale, noise addition
Model Utilities	ModelCheckpoint, EarlyStopping, ReduceLROnPlateau	For safe training and improving convergence



Steps to execute code
Data Preprocessing in Google Earth Engine
The study utilized Sentinel-2 Multispectral Instrument (MSI) Level-2A surface reflectance collection imagery derived from the Copernicus Open Access Hub, hosted in Google Earth Engine (GEE), catalog identifier COPERNICUS/S2\_SR\_HARMONIZED. The Sentinel-2 data set, with 13 spectral bands in the visible, near-infrared, and short-wave infrared ranges and spatial resolutions of 10 m - 60 m, is suitable for land-use and vegetation differentiation in mixed urban landscapes. Imagery was acquired during 2023, and new in situ samples were manually labeled based on ground-truthing in the 2024 calendar year for inter-annual comparison and compatibility. The area of interest (AOI) is the Perungudi area of Chennai, India (80.2305 E - 80.2605 E, 12.9605 N - 12.9805 N)

Manual Annotation and Ground-Truth Creation
Furthermore, four spatially diverse tiles measuring 256 pixels by 256 pixels (almost 2.56 km by 2.56 km each) were extracted to denote urban, water, vegetation, and wasteland land covers based on the centroid locations manually collected for the region. They were subsequently annotated using polygon digitization in GEE for the reference year of 2024, and then exported as Geo-TIFFs for model training and evaluation. Polygon digitization was conducted for the 2024 Sentinel-2 composite within GEE where the polygons were comprehensive representation of high quality reference, and classified as Urban, Vegetation, Water Body and Wasteland Land Covers, through exerting quality control of polygons with high- resolution Google Earth imagery and municipal Cadastral maps. The reference dataset provides the basis for supervised training and validation for the comparative Random Forest and IRUNet-based classification models described in the following sections.

Models Mentioned in Notebook
Based on the visible code:
•	U-Net
•	U-Net++ / Nested U-Net
•	ATTENTION U-Net
•	DeepLabV3+
•	SegNet
•	Custom CNN-based segmentation mode
•	Test-Time Augmentation (TTA)
•	Visualisation utilities
•	Model comparison and best model selection logic

Steps to Execute the Code
STEP 1 — Install Required Libraries
Run this once:
pip install tensorflow opencv-python matplotlib numpy seaborn scikit-learn imgaug

STEP 2 — Mount Dataset / Set Paths
Your notebook expects folders such as:
/Dataset/images/
/Dataset/masks/
Make sure your dataset is placed correctly and update the path variables:
IMAGE\_PATH = "/content/Dataset/images"
MASK\_PATH = "/content/Dataset/masks"

STEP 3 — Import All Libraries
Your code starts with:
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

STEP 4 — Load and Preprocess the Dataset
This step includes:
•	Reading images and segmentation masks
•	Resizing them to a fixed size (e.g., 256×256)
•	Normalizing pixel values
•	Converting masks to one-hot encoded labels
•	Splitting dataset into train/validation
Example:
X\_train, X\_val, y\_train, y\_val = train\_test\_split(images, masks, test\_size=0.2)
STEP 5 — Data Augmentation
Your notebook performs:
•	Random flips
•	Rotation
•	Scaling
•	Brightness change
•	Noise addition
Example:
X\_train\_aug, y\_train\_aug = augment\_dataset(X\_train, y\_train)

STEP 6 — Build Segmentation Models
For example:
model = build\_unet(input\_shape=(256,256,3), num\_classes=4)
Other models trained:
•	U-Net
•	UNet++
•	SegNet
•	DeepLabV3+
•	Attention U-Net
STEP 7 — Compile the Model
Common compiler settings:
model.compile(optimizer="adam",
loss="categorical\_crossentropy",
metrics=\["accuracy"])

STEP 8 — Train Safely with Callbacks
The notebook uses:
•	EarlyStopping
•	ReduceLROnPlateau
•	ModelCheckpoint
Example:
history = train\_model\_safely(model, X\_train\_aug, y\_train\_aug, X\_val, y\_val)

STEP 9 — Visualize Training Results
Your notebook plots:
•	Loss curves
•	Accuracy curves
•	Predicted vs True masks
visualize\_results(model, X\_val, y\_val, model\_name)

STEP 10 — Perform Test-Time Augmentation (TTA)
You use TTA for stable prediction:
pred\_tta = test\_time\_augmentation(model, X\_val\[0])

STEP 11 — Compare Models and Pick Best One
The code checks validation accuracy:
best\_model = max(trained\_models, key=lambda name: model\_accuracy\[name])





Dataset description
The level 2A surface reflectance dataset used in this study is taken from the Sentinel-2 Multispectral Instrument (MSI) (Copernicus Open Access Hub) and is available through Google Earth Engine (GEE). The collection ID:{COPERNICUS/S2\_SR\_HARMONIZED}
The Sentinel-2 satellite has 13 spectral bands in the visible, near-infrared, and short-wave infrared, with a spatial resolution of 10 to 60 meters that can be used for land-use and vegetation classifications in a heterogeneous urban environment. Images were downloaded according to the 2023 time period, and were subsequently validated against manually annotated samples that were collected in 2024 to ensure that there was an adequate degree of agreement across the inter-annual time period being examined. The area of interest (AOI) was delineated as the Perungudi area of Chennai, India with geographic coordinates of (80.2305 E - 80.2605 E, 12.9605 N - 12.9805 N). The AOI is an important land transition corridor located near the Chennai IT Expressway and, as such, has ongoing vegetation to built-up cover transitions that were captured in past years.



Results summary



The Sentinel-2 imagery for the Perungudi region was prepared for deep learning analysis using a structured workflow. Raw multi-band GeoTIFF files were ingested and preprocessed to normalize values across spectral channels. The imagery was then partitioned into over 1,000+ spatial segments, and further divided into 128×128 pixel patches, each containing all relevant spectral bands. Data augmentation and normalization were performed to enhance robustness and facilitate training stability. Ground-truth masks for each partition, provided as co-registered raster files, served as pixel-wise supervision for model training and validation. The core segmentation engine was an IRUNet, constructed by integrating an InceptionResNetV2 encoder (initialized with ImageNet weights, excluding classification layers) with a UNet-style multi-scale decoder. Skip connections were established between selected intermediate layers in the encoder and decoder, enabling the model to retain fine contextual and boundary details during upsampling. Categorical cross-entropy loss was used for multi-class segmentation, and the model was optimized with the Adam algorithm. Evaluation metrics included per-class accuracy and mean Intersection-over-Union (IoU), calculated across all partitions. Three UNet-based architectures were implemented and tested: a standard UNet, a ResUNet variant, and the IRUNet configuration. Although full end-to-end model training was restricted by resource constraints, preliminary results demonstrate that the IRUNet achieved an overall accuracy of 0.97. Output segmentation maps and overlays confirm effective classification of urban, water, and vegetation classes, with successful separation of complex spatial boundaries and mixed pixels. This scalable workflow is designed to automate result collation into a database for subsequent interactive dashboard analytics and further validation.



Youtube demo link :https://youtu.be/0dWCLqOtHwU

