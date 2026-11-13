# ASD_Frequency_Domain_Classification
MATLAB scripts for frequency-domain feature extraction, feature selection, and fuzzy ranking in ASD classification


This repository contains MATLAB scripts developed for the study **"Fuzzy-Interpretable Local Time–Frequency Features in Resting-state fMRI for Autism Spectrum Disorder."**

The scripts implement key steps of the proposed frequency-domain analysis and classification framework for autism spectrum disorder (ASD) using resting-state functional MRI (rs-fMRI) data.

 Repository Contents

The repository currently includes the following MATLAB scripts:

 1. `LocalFeatureExtraction.m`

This script performs frequency-domain feature extraction from independent component time series using the Continuous Wavelet Transform (CWT). It extracts features across predefined frequency bands and calculates statistical and signal-complexity descriptors.

 2. `LocalfeatureSelectionClassificationLOO.m`

This script implements the feature-selection and classification procedure using a nested cross-validation framework. It incorporates outer leave-one-out cross-validation (LOOCV), inner stratified five-fold cross-validation, greedy forward feature selection, and a linear support vector machine (SVM) classifier.

 3. `LocalPostFuzzy.m`

This script performs the post-processing and fuzzy evaluation stage of the framework. It applies the Mamdani fuzzy inference approach to integrate the relevant feature-ranking criteria and obtain fuzzy importance scores.

 Methodological Overview

The overall computational framework consists of three main stages:

1. **Frequency-domain feature extraction:** CWT-based analysis and calculation of signal descriptors across predefined frequency bands.
2. **Feature selection and classification:** Nested cross-validation with greedy forward feature selection and linear SVM classification.
3. **Fuzzy evaluation:** Mamdani fuzzy inference for integrating multiple evaluation criteria and obtaining feature-importance scores.

 Requirements

- MATLAB
- MATLAB toolboxes required by the individual scripts, including the Statistics and Machine Learning Toolbox and, where applicable, the Wavelet Toolbox.

Please ensure that the required MATLAB toolboxes are installed before running the scripts.

 Data Availability

The scripts are provided for methodological transparency and reproducibility. The participant-level rs-fMRI data are not included in this repository. Access to the original dataset is subject to the data-sharing policies and conditions of the respective data provider.

 Usage

Download or clone this repository and open the MATLAB scripts in MATLAB. Before execution, review the input and output directory settings in each script and update the paths according to your local environment.

The scripts may require appropriate input data and intermediate files generated during the preceding processing steps.

 Citation

If you use these scripts in your research, please cite the associated publication:

*Haghighat, H. (Year). Fuzzy-Interpretable Local Time–Frequency Features in Resting-state fMRI for Autism Spectrum Disorder. [Journal and publication details to be added after publication.]*

 Contact

For questions regarding the implementation, please contact the repository author through GitHub.
