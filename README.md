## Introduction

The image and data files were provided to me by Sarah Pyfrom. Data was generated using CellProfiler and will be documented on a future date. The purpose of this repo is to take tiff images and convert them into quantified data that enable us to characterize cells with xist clouds.



## Goals

I envision that this repo will have three modules.

1. Feature extraction. For now, this will be handled by the CellProfiler gui for now. It is a stretch goal for my lab rotation to automate this step so that we have more control over noise filtering and anomaly exclusion.

2. Image classification. I have various ideas how this would work. If there is clear clustering of features for classifying an xist cloud as being localized, dispersed, or diffused, we can set some hard cutoffs. Otherwise, we could try a multiclass classifier, such as logistic regression or SVM. If the features are less clustered and more continuous, we could instead plot on a histogram or violin plots.

3. Troubleshooting. This will be important in evaluating the accuracy of the classification. For this, I envision that for each microscope field, we would output one image per cell labeled by cell_number in labeled directories indicating how that cell was classified. For example, if we had four cells classified into three groups, this will be the output.

   ```
   images
   └── output
       ├── type_1
       │   ├── sSP67_18_B6_CRE_003-cell_1.png
    	│   └── sSP67_18_B6_CRE_003-cell_3.png
       ├── type_2
       │   └── sSP67_18_B6_CRE_003-cell_4.png
       └── type_4
    	    └── sSP67_18_B6_CRE_003-cell_2.png
   ```

   In this way, it makes it easy for a user to scroll through a repository and identify if a cell was misclassified.



## Miscellaneous Notes

To save Bokeh figures, you must install geckodriver:

```
sudo apt-get install firefox-geckodriver
```

