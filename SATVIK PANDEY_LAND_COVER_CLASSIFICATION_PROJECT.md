# Land Use and Land Cover Classification of Southern India

## 1. Overview

This research project compares traditional machine learning and deep learning approaches for land use and land cover (LULC) classification across Southern India. The study uses multi-spectral Landsat 8 imagery and ESA WorldCover v200 reference labels.

The project implements:

- A Google Earth Engine pipeline for regional data preparation and change detection.
- A traditional machine-learning pipeline using engineered spectral features.
- A deep-learning pipeline using 32Ã—32 multi-spectral image patches.
- A two-stage hierarchical classification strategy.
- Confusion-matrix, uncertainty, area-discrepancy, and map-generation analyses.

## 2. Research Summary

| Model | Input | Accuracy | Macro F1-score | Relative cost |
| --- | --- | ---: | ---: | --- |
| Stacking ensemble | Engineered tabular features | 82.88% | 0.78 | Low to moderate CPU cost |
| EfficientNetB0 | 32Ã—32 multi-spectral patches | 90.04% | Not consistently reported | High GPU cost |
| EfficientNetB0, final evaluation | 32Ã—32 multi-spectral patches | 91.16% | 0.84 | High GPU cost |
| Level-1 hierarchical model | Broad land-cover patches | 96.41% | Not consistently reported | High GPU cost |

The adapted EfficientNetB0 model achieved the best reported overall accuracy of **91.16%**. The Level-1 hierarchical model reached **96.41%** accuracy for broad categories. The traditional ensemble remains useful when feature engineering is available and GPU resources are limited.

> **Important:** Results come from different experiments and class definitions. The reported metrics should be compared with caution rather than treated as a single controlled benchmark.

## 3. Project Objectives

1. Build reproducible LULC pipelines for Southern India.
2. Compare traditional ML and deep learning under the same study region.
3. Reduce spectral confusion between similar vegetation classes.
4. Evaluate the effect of class merging and focal loss on minority classes.
5. Generate spatial prediction, confidence, uncertainty, and change maps.
6. Produce technical documentation and reproducible model-training artifacts.

## 4. Study Area and Datasets

### 4.1 Study Area

- Region bounds: **72, 8, 88.5, 23**
- Coordinate format: **[longitude_min, latitude_min, longitude_max, latitude_max]**
- Spatial extent: Southern India

### 4.2 Landsat 8 Data

- Dataset: **Landsat 8 Collection 2 Tier 1 Surface Reflectance**
- Spatial resolution: **30 m**
- Temporal coverage: annual median composites for **2000, 2010, and 2020**
- Bands used:
  - Blue
  - Green
  - Red
  - Near-Infrared (NIR)
  - Short-wave Infrared 1 (SWIR1)
  - Short-wave Infrared 2 (SWIR2)

The GEE pipeline performs QA_PIXEL cloud and cloud-shadow masking before forming median composites.

### 4.3 ESA WorldCover Labels

- Dataset: **ESA WorldCover v200**
- Spatial resolution: **10 m**
- Reference year: **2020**
- Purpose: supervised LULC labels and validation reference

## 5. Class Definitions

The final deep-learning experiments use these primary classes:

| Class ID | Class | Description |
| ---: | --- | --- |
| 20 | Forest | Dense forest vegetation |
| 30 | Mixed Vegetation | Merged shrubland, grassland, and cropland |
| 50 | Urban | Built-up and impervious surfaces |
| 60 | Bare land | Exposed soil, rocks, and non-vegetated ground |
| 70 | Water | Water bodies and open water |
| 90 | Coastal/Aquatic Vegetation | Merged wetlands and mangroves |

The following class merges were applied:

- Shrubland + Grassland + Cropland â†’ **Mixed Vegetation**
- Wetlands + Mangroves â†’ **Coastal/Aquatic Vegetation**

This strategy reduced spectral ambiguity and addressed the observed zero-recall problem for rare classes.

## 6. Methodology

### 6.1 Google Earth Engine Baseline

The GEE workflow includes:

1. Defining the Southern India study region.
2. Filtering Landsat images by date and bounds.
3. Applying QA_PIXEL cloud and cloud-shadow masks.
4. Converting Level-2 values to surface reflectance.
5. Creating median composites for 2000, 2010, and 2020.
6. Adding NDVI as an additional image band.
7. Sampling labelled pixels from ESA WorldCover.
8. Training a Random Forest classifier.
9. Producing 2000, 2010, and 2020 LULC maps.
10. Computing a change map between 2000 and 2020.
11. Exporting results to Google Drive.

The GEE baseline uses:

- **5 input features:** Blue, Green, Red, NIR, and NDVI
- **Random Forest:** 100 trees
- **Training pixels:** 5,000
- **Classification scale:** 30 m

### 6.2 Traditional Machine Learning

The feature-based pipeline computes spectral and contextual features including:

- NDVI
- NDBI
- NDWI
- EVI and related vegetation indices
- NDVI amplitude
- Spectral standard deviation
- Local neighborhood statistics

The final ensemble contains:

- XGBoost
- LightGBM
- CatBoost
- Logistic Regression meta-learner

Cross-validation uses **5-fold stratified validation** with a fixed random seed of **42**.

### 6.3 Deep Learning

The deep-learning pipeline uses:

- Patch size: **32Ã—32 pixels**
- Spectral channels: **6 Landsat channels**
- Optional additional channels: engineered spectral indices
- Backbone: **EfficientNetB0**
- Input adaptation: 1Ã—1 convolution to project the multi-spectral input into three channels
- Optimizer: Adam
- Learning rate: **1e-3** or **1e-4**
- Loss functions: sparse categorical cross-entropy, focal loss, and class-weighted loss
- Data augmentation: horizontal flip, vertical flip, and rotation
- Callbacks: ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau

The model learns spatial and spectral patterns from neighbouring pixels while preserving the original multi-spectral information through the custom projection layer.

### 6.4 Hierarchical Classification

The hierarchical design has two levels:

1. **Level 1:** Vegetation, Urban, Bare land, and Water.
2. **Level 2:** Forest, Mixed Vegetation, and Coastal/Aquatic Vegetation.

This structure separates broad classes first and then performs fine-grained classification within vegetation classes.

## 7. Training and Evaluation Configuration

| Setting | Value |
| --- | ---: |
| Training split | Typically 80% training and 20% test |
| Experimental split | 70% training, 15% validation, 15% test |
| Validation strategy | Stratified splitting |
| Optimizer | Adam |
| Primary loss | Sparse categorical cross-entropy |
| Class imbalance handling | Class weighting and focal loss |
| Patch size | 32Ã—32 pixels |
| Batch size | Notebook-dependent |
| Early stopping | Enabled |
| Model checkpoint | Best weights saved |
| Random seed | 42 |
| Preferred hardware | NVIDIA GPU with 6â€“12 GB VRAM or more |

## 8. Results

### 8.1 Overall Performance

| Model | Overall Accuracy | Macro F1-score |
| --- | ---: | ---: |
| Traditional stacking ensemble | **82.88%** | **0.78** |
| EfficientNetB0 | **90.04%** | Not consistently reported |
| EfficientNetB0, final evaluation | **91.16%** | **0.84** |
| Level-1 hierarchical model | **96.41%** | Not consistently reported |

The best reported deep-learning result is **91.16%** overall accuracy across the primary land-cover classes.

### 8.2 Final EfficientNetB0 Classification Report

The final evaluation contains **7,968 test patches**.

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| Forest | 0.72 | 0.39 | 0.51 | 806 |
| Mixed Vegetation | 0.92 | 0.97 | 0.94 | 6,385 |
| Urban | 0.69 | 0.67 | 0.68 | 98 |
| Bare land | 0.80 | 0.67 | 0.73 | 140 |
| Water | 0.89 | 0.97 | 0.93 | 533 |
| Wetlands | 0.00 | 0.00 | 0.00 | 1 |
| Mangroves | 0.00 | 0.00 | 0.00 | 5 |
| **Overall accuracy** | â€” | â€” | **91.16%** | **7,968** |
| **Macro average** | 0.57 | 0.52 | **0.54** | 7,968 |
| **Weighted average** | 0.89 | 0.90 | **0.89** | 7,968 |

The extremely small support for Wetlands and Mangroves means their zero scores are statistically unreliable. They should not be interpreted as proof that those classes are impossible to classify.

### 8.3 Main Findings

- The EfficientNetB0 model improved overall accuracy by approximately **8.28 percentage points** over the traditional ensemble.
- The Level-1 hierarchical model reached **96.41%** accuracy for broad classes.
- Forest and Mixed Vegetation had the highest confusion.
- Deep learning produced smoother spatial predictions than pixel-level ML.
- Class imbalance remained the main limitation for rare classes.
- Focal loss and class weighting reduced but did not eliminate poor minority-class performance.
- 30 m Landsat resolution limits fine boundary detection.

## 9. Generated Outputs

The notebook produces:

1. Study-area overview map.
2. Hierarchical classification framework diagram.
3. Training and validation accuracy curves.
4. Training and validation loss curves.
5. Final confusion matrix.
6. Per-class area discrepancy chart.
7. Spatial entropy or uncertainty map.
8. Thresholded confidence map.
9. Final predicted LULC patch map.
10. 2000 LULC map.
11. 2010 LULC map.
12. 2020 LULC map.
13. 2000â€“2020 change map.

## 10. Recommended Environment

### 10.1 Python

Use Python **3.10 or 3.11**.

### 10.2 Dependencies

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
pip install xgboost lightgbm catboost rasterio geopandas
pip install earthengine-api opencv-python tqdm imbalanced-learn
```

### 10.3 Hardware

| Component | Minimum | Recommended |
| --- | --- | --- |
| RAM | 16 GB | 32 GB |
| Storage | 50 GB | 150 GB |
| CPU | 4 cores | 8+ cores |
| GPU | NVIDIA GPU with 6 GB VRAM | 12+ GB VRAM |

A CUDA-capable GPU is strongly recommended for deep-learning experiments.

## 11. Reproduction Guide

### 11.1 Google Drive Setup

Run the notebook setup cell and mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 11.2 Execution Order

1. Install the required Python libraries.
2. Mount Google Drive.
3. Run the Google Earth Engine export script.
4. Verify the exported image tiles and labels.
5. Extract 32Ã—32 image patches.
6. Normalize the input features.
7. Create training, validation, and test sets.
8. Train the traditional ML ensemble.
9. Train the EfficientNetB0 model.
10. Run the Level-1 and Level-2 hierarchical experiments.
11. Generate classification reports and confusion matrices.
12. Produce uncertainty, confidence, and map outputs.
13. Save all checkpoints and model configuration metadata.

## 12. Reproducibility Notes

- Use the same class mapping for every experiment.
- Keep the random seed fixed at **42**.
- Lock the training, validation, and test partitions when comparing models.
- Save normalization parameters and label mappings.
- Record model hyperparameters, batch size, hardware, and training duration.
- Report per-class support values along with every metric.
- Do not compare results that use different class merges or data splits.

## 13. Limitations

- Wetlands and Mangroves have extremely limited samples.
- The 30 m Landsat resolution can reduce boundary accuracy.
- Class imbalance can bias precision, recall, and F1-score.
- Results are based on a limited set of experimental runs.
- Different class definitions and split configurations reduce direct comparison.
- ESA WorldCover labels may contain boundary uncertainty.
- The models should be validated with independent field data or high-resolution imagery.
- The project does not currently provide a production inference API or web interface.

## 14. Recommended Future Work

1. Add multi-temporal Landsat or Sentinel-2 imagery for seasonal classification.
2. Fuse Sentinel-1 SAR data with optical imagery.
3. Use active learning or synthetic data generation for rare classes.
4. Compare EfficientNetB0 with EfficientNetV2, ConvNeXt, ResNet, and Vision Transformer models.
5. Add test-set calibration and uncertainty estimation.
6. Evaluate with balanced metrics across regions and years.
7. Add a semi-supervised or self-supervised pretraining stage.
8. Develop a repeatable inference pipeline for new satellite scenes.
9. Publish model weights with a versioned training configuration.
10. Validate outputs using field surveys and higher-resolution reference data.

## 15. Project Requirements Summary

| Requirement | Minimum |
| --- | --- |
| Python | 3.10 |
| RAM | 16 GB |
| Disk space | 50 GB |
| GPU | Optional for inference, recommended for training |
| Google Account | Required for GEE Drive export |
| Google Earth Engine Access | Required for dataset preparation |
| Internet Access | Required for downloading datasets and packages |

## 16. Citation

If this project is used in a research or academic publication, cite the notebook and clearly state the dataset versions, study region, class definitions, and model parameters used in the reported experiment.

## 17. License

This project does not currently declare a repository license. All third-party datasets, libraries, and model assets must be used according to their original terms and licenses.

## 18. Contributions

Contributions, bug reports, and improvements are welcome. All changes should preserve reproducibility and document any modifications to preprocessing, class definitions, training configuration, or evaluation metrics.

---

# Research Paper Report

## 1. Paper Title

**Hierarchical Land Use Land Cover Classification Using Sentinel-2A Satellite Imagery**

## 2. Authors

- Prekshita Singh Tanwar
- Satvik Pandey
- Venushree Gayathri Chittari Uddanti
- Mahaswetha S
- Anand Pal
- Ujjwal Verma

## 3. Abstract

Land use and land cover classification is essential for environmental monitoring, urban planning, and resource allocation. This paper proposes a two-stage hierarchical classification framework that combines broad ecological grouping with fine-grained land-cover prediction. Sentinel-2A imagery from four geographically diverse regions of India is used to evaluate cross-region generalization.

The proposed pipeline uses a 114-channel pixel-level feature representation comprising:

- 13 raw spectral bands,
- 10 spectral indices, and
- localized neighborhood statistics calculated with 15×15 sliding windows.

The hierarchical Random Forest + Multi-Layer Perceptron configuration achieved an overall accuracy of **0.85** and a macro F1-score of **0.64**. This represents a **22 percentage-point improvement in overall accuracy** over the corresponding flat Random Forest baseline. The hierarchical framework also demonstrated improved minority-class performance and reduced class confusion.

**Keywords:** Land Use and Land Cover Classification, Machine Learning, Sentinel-2, Remote Sensing, Urban Planning.

## 4. Research Contributions

1. A two-tier hierarchical classification system that separates broad ecological groups before fine-grained prediction.
2. A 114-channel feature space containing raw bands, spectral indices, and local neighborhood statistics.
3. Cross-region evaluation using northern, southern, western, and eastern Indian regions.
4. Comparison with flat classifiers, stacking ensembles, CNNs, EfficientNetB0, and several transformer-based models.
5. Uncertainty analysis using entropy maps and confidence thresholding.

## 5. Study Regions

The paper evaluates four geographic regions:

| Region | Approximate location |
| --- | --- |
| North India | 30°N–33°N, 75°E–78°E |
| South India | 9°N–12°N, 78°E–81°E |
| West India | 21°N–24°N, 69°E–72°E |
| East India | 18°N–21°N, 84°E–87°E |

The primary dataset also includes areas in Punjab, Tamil Nadu, Odisha, Gujarat, Madhya Pradesh, Chhattisgarh, and other selected Indian regions.

## 6. Data Sources

| Dataset | Resolution | Purpose |
| --- | ---: | --- |
| Sentinel-2A Level-2A | 10, 20, and 60 m | Multispectral feature extraction |
| Landsat 8 Collection 2 Tier 1 | 30 m | Additional comparison dataset |
| ESA WorldCover v200 | 10 m | Reference labels |
| EuroSAT RGB benchmark | 10 m | Deep-learning baseline comparison |

The Sentinel-2A bands include B01–B07, B8A, B11, and B12, together with the Scene Classification Layer. The raw values were scaled by 10,000 before normalization.

## 7. Feature Engineering

### 7.1 Spectral Indices

The paper uses the following indices:

- NDVI: vegetation vigor and biomass.
- NDBI: built-up surface identification.
- NDWI: water-body detection.
- NDMI: canopy moisture content.
- BSI: bare-soil detection.
- EVI2: vegetation monitoring with reduced atmospheric effects.
- GNDVI: chlorophyll-sensitive vegetation assessment.
- MNDWI: improved water extraction using SWIR.
- AWEI: water extraction with shadow and dark-surface handling.
- SI: soil and shadow differentiation.

### 7.2 Spatial Features

A 15×15 neighborhood was processed with depth-wise convolution in TensorFlow. For every pixel, the pipeline extracted:

- Local mean,
- Local standard deviation,
- Local minimum,
- Local maximum, and
- Difference from the local mean.

The complete representation therefore contains **114 channels per pixel**.

### 7.3 Preprocessing

1. Reproject all labels and image data to a common coordinate reference system.
2. Resample features to a common 20 m grid.
3. Scale spectral values from 0 to 1 through min-max normalization.
4. Use nearest-neighbor interpolation for categorical label alignment.
5. Apply stratified training, validation, and test splitting.
6. Use minority-class oversampling and spectral augmentation.

## 8. Classification Schema

The original WorldCover labels are mapped into the following 10 classes:

| Index | WorldCover ID | Class |
| ---: | ---: | --- |
| 0 | 10 | Tree Cover |
| 1 | 20 | Shrubland |
| 2 | 30 | Grassland |
| 3 | 40 | Cropland |
| 4 | 50 | Built-up |
| 5 | 60 | Sparse Vegetation |
| 6 | 80 | Permanent Water Bodies |
| 7 | 90 | Herbaceous Wetland |
| 8 | 95 | Mangroves |
| 9 | 100 | Moss/Lichen |

The paper’s hierarchical grouping is:

| Coarse group | WorldCover classes |
| --- | --- |
| Vegetation | Trees, Shrubland, Grassland |
| Cropland | Cropland |
| Urban | Built-up |
| Barren | Sparse Vegetation |
| Water/Wetlands | Water, Wetland, Mangroves |

The default ecological hierarchy uses a coarse classifier followed by a fine classifier. For the vegetation-specialist deep-learning experiment, the final classes are Forest, Mixed Vegetation, Urban, Bare land, Water, and Coastal/Aquatic Vegetation.

## 9. Experimental Models

### 9.1 Flat Baselines

- Random Forest
- Multi-Layer Perceptron
- Tabular Transformer
- CNN
- EfficientNetB0
- Stacking ensemble using XGBoost, LightGBM, and CatBoost

### 9.2 Hierarchical Models

| Configuration | Coarse model | Fine model |
| --- | --- | --- |
| Hierarchical XGB | Random Forest | XGBoost |
| Hierarchical MLP | Random Forest | MLP |
| Hierarchical ResNet-MLP | Random Forest | ResNet-MLP |
| Hierarchical FT-Transformer | Random Forest | Feature Tokenizer Transformer |
| Hierarchical Tabular Transformer | Random Forest | Tabular Transformer |
| Two-Tier Tabular Transformer | Tabular Transformer | Tabular Transformer |
| MLP + Tabular Transformer | MLP | Tabular Transformer |
| DL-EfficientNetB0 Hierarchy | EfficientNetB0 | EfficientNetB0 |

All models were trained with TensorFlow 2.15 and Scikit-Learn. The Random Forest baseline uses 100 estimators, a maximum depth of 10, minimum 50 samples per leaf, sqrt feature sampling, and balanced class weights.

## 10. Deep-Learning Hyperparameters

| Component | Configuration |
| --- | --- |
| CNN | Conv2D(32, 3×3), MaxPool, Conv2D(64, 3×3), MaxPool, Dense(64), Softmax(10) |
| Image size | 64×64 pixels |
| EfficientNetB0 | ImageNet-pretrained; fine-tuned classification head |
| MLP | Dense(128), Dense(64), ReLU, Dropout(0.5), Batch Normalization |
| ResNet-MLP | Dense(64)-Dense(128)-Dense(64)-Dense(64) with residual connections |
| Tabular Transformer | Two transformer blocks; four attention heads; 64-dimensional representation |
| Feature Tokenizer Transformer | 32-dimensional token representation; two transformer blocks |
| Stacking ensemble | XGBoost, LightGBM, CatBoost with Logistic Regression meta-learner |
| Cross-validation | Stratified 5-fold |

## 11. Evaluation Metrics

The paper evaluates models with:

- Overall Accuracy (OA),
- Macro F1-score,
- Weighted F1-score,
- Per-class precision, recall, and F1-score,
- Confusion matrices,
- Proportional Area Bias, and
- Shannon entropy uncertainty.

The area-bias metric is:

$$
\text{Area Bias} = \frac{\text{Predicted Area} - \text{Reference Area}}{\text{Reference Area}} \times 100\%.
$$

The entropy metric is:

$$
H = -\sum_{c=1}^{C} p_c \log(p_c).
$$

Pixels with a maximum probability below 0.7 are flagged as uncertain and assigned class 255.

## 12. Results

### 12.1 Model Comparison

| Model | OA | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| Random Forest | 0.63 | 0.24 | 0.66 |
| MLP | 0.53 | 0.36 | 0.52 |
| Tabular Transformer | 0.54 | 0.34 | 0.54 |
| Stacking Ensemble | 0.82 | — | 0.78 |
| CNN Baseline | 0.65 | 0.61 | 0.63 |
| EfficientNetB0 | 0.82 | 0.60 | 0.76 |
| RF + XGBoost | 0.85 | 0.62 | 0.84 |
| RF + MLP | **0.85** | **0.64** | **0.85** |
| RF + ResNet-MLP | 0.84 | 0.63 | 0.84 |
| RF + Tabular Transformer | 0.84 | 0.63 | 0.84 |
| RF + FT-Transformer | 0.85 | 0.63 | 0.84 |
| XGBoost + Tabular Transformer | 0.81 | 0.54 | 0.79 |
| Two-Tier Tabular Transformer | 0.83 | 0.62 | 0.83 |
| DL-EfficientNetB0 (Vegetation Specialist) | **0.91** | **0.54** | **0.89** |

The RF + MLP configuration achieved the best overall performance under the common ecological hierarchy. Its overall accuracy was **0.85**, and its macro F1-score was **0.64**.

### 12.2 Improvement over the Flat Baseline

The RF + MLP model produced:

- **22 percentage points of improvement in overall accuracy** over the flat Random Forest baseline,
- **40 percentage points of improvement in macro F1-score**, and
- Improved detection of Cropland, Sparse Vegetation, and Water.

The best deep-learning model achieved **91% overall accuracy**, but its macro F1-score was only **0.54** because of class imbalance and the different vegetation-specialist label schema.

### 12.3 Important Findings

- The hierarchical framework made a larger contribution than the downstream classifier.
- XGBoost + Tabular Transformer performed poorly, indicating that coarse-level classification strongly affected final performance.
- Tree and Water classes were generally stable.
- Built-up classification showed a precision-recall trade-off: precision was 0.42 and recall was 0.77.
- Shrubland was difficult to classify because of its small support.
- Wetland recall remained limited, with much of the Wetland area being confused with Open Water.
- Grassland, Built-up, and Wetland classes had the largest area-bias errors.

## 13. Feature-Ablation Results

The RF + MLP configuration was evaluated using four feature sets over 10 random seeds.

| feature Set | Macro F1 | Accuracy | ENR |
| --- | ---: | ---: | ---: |
| Spectral bands only | 0.461 ± 0.013 | 0.769 ± 0.031 | 0.031 ± 0.004 |
| Bands + indices | 0.486 ± 0.009 | 0.776 ± 0.017 | 0.030 ± 0.002 |
| Bands + indices + neighborhood statistics | 0.573 ± 0.007 | 0.856 ± 0.006 | 0.019 ± 0.001 |
| Final representation + oversampling | **0.627 ± 0.007** | **0.840 ± 0.005** | **0.024 ± 0.001** |

The Friedman test found statistically significant differences among all four configurations at $p < 0.001$. Neighborhood statistics produced the largest statistically significant improvement, while oversampling significantly improved macro recall and macro F1-score.

## 14. Uncertainty Analysis

The entropy map showed:

- Low uncertainty in homogeneous water and dense-vegetation areas.
- High uncertainty at land-cover boundaries and transitional zones.
- Confident predictions in Punjab for Trees, Shrubland, Cropland, and Built-up classes.
- More scattered uncertainty in the Puducherry study region because of coastline and mixed land-cover transitions.

The confidence thresholding analysis used a maximum-probability threshold of **0.70**. Pixels below this threshold were flagged as uncertain.

## 15. Spatial Results

The paper analyzes Puducherry and Punjab:

- **Puducherry:** The predicted map showed a clear separation between water and coastal land.
- **Punjab:** The predicted map showed a wide distribution of Cropland and vegetation.

These results support cross-region generalization, although local accuracy remained dependent on class representation and spectral diversity.

## 16. Limitations and Research Gaps

1. The study uses a limited number of geographic regions and does not provide complete nationwide validation.
2. Rare classes, including Shrubland, Wetlands, and Mangroves, have very small support.
3. The EfficientNetB0 hierarchy uses a different label schema from the flat RF-based models.
4. The macro F1-score remains substantially lower than the overall accuracy for highly imbalanced datasets.
5. The model has difficulty distinguishing homogeneous classes near boundaries.
6. The pixel-level approach does not directly use temporal information.
7. The study does not provide a production inference system or web application.
8. The results should be validated with independent field surveys or higher-resolution reference datasets.

## 17. Discussion

The results show that hierarchical decomposition is more important than the choice of a particular downstream classifier. The first stage creates more separable super-groups, while the second stage focuses only on classes relevant to each super-group. This reduces error propagation and improves minority-class recognition.

The paper also shows that deep-learning models can learn spatial patterns, but their macro-F1 performance is limited by class imbalance. Classical models remain useful when feature engineering and limited computational resources are important.

## 18. Conclusion

The proposed hierarchical framework substantially improves land-cover classification performance compared with flat baselines. The RF + MLP model achieved an overall accuracy of **0.85** and a macro F1-score of **0.64**, while the vegetation-specialist EfficientNetB0 model achieved **0.91** overall accuracy.

However, the reported high accuracy should be interpreted together with macro F1-score and per-class support. The study demonstrates that hierarchical learning, spectral indices, local neighborhood features, and class balancing are useful for remote-sensing classification. Future work should focus on multi-temporal data, transformer models, higher-resolution imagery, uncertainty calibration, active learning, and deployment-oriented inference pipelines.

## 19. Publication-Specific Notes

- The paper uses **Sentinel-2A** for the main methodology and **Landsat 8** for comparative data preparation.
- The paper’s main public dataset is **ESA WorldCover v200**.
- The paper uses a 10-class schema, but the final vegetation-specialist experiment removes Moss/Lichen and merges vegetation species into broader groups.
- Some reported deep-learning results use different labels and class support; they should therefore not be compared directly with the RF + MLP 10-class results.
- The current project’s notebook uses a Southern India Landsat 8 workflow and therefore represents a related but distinct experimental configuration.

## 20. Research Reproducibility Checklist

1. Obtain Sentinel-2A Level-2A scenes from Copernicus.
2. Collect ESA WorldCover v200 labels.
3. Reproject all rasters to a common CRS and spatial grid.
4. Extract raw bands and spectral indices.
5. Compute 15×15 neighborhood statistics.
6. Create stratified train, validation, and test sets.
7. Train the flat and hierarchical models with fixed random seeds.
8. Save class mappings, normalization parameters, and model checkpoints.
9. Generate confusion matrices and performance reports.
10. Produce entropy, confidence, and final LULC maps.
11. Report class support and macro metrics alongside overall accuracy.
12. Validate results with independent field or high-resolution reference data.

## 21. References

The paper’s bibliography is maintained in the project’s `references.bib` file. The bibliography should be compiled with Biber and BibLaTeX when the LaTeX source is published.

---

## Research Scope Note

The **Research Paper Report** describes the paper attached in the user request. The **Project Overview, Dataset, Environment, and Reproduction Guide** sections describe the current Southern India Landsat 8 and EfficientNet notebook. These two experimental configurations are related but not identical, and their results must not be treated as a single controlled benchmark.
