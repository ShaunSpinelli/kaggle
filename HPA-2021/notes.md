# Notes

Human Proteins Atlas aimed at mapping proteins in human cells.

for each train image given mulitple labels

labels for each image are in a CSV

each sample contans 4 images

-  red : microtuble channels
-  blue: nuclei channels
-  yellow: Endoplasmic Reticulum
-  green: protien of inerrest (cooresponds to labels for image)


We are doining in stance segmentation on by cell then applying labels to each segemnt.

## Submission File 


```
ImageID,ImageWidth,ImageHeight,PredictionString
ImageAID,ImageAWidth,ImageAHeight,LabelA1 ConfidenceA1 EncodedMaskA1 LabelA2 ConfidenceA2 EncodedMaskA1

```


labels =  by image segmentation with  0 - N classes

predictions =  per instance of cell with 0-N classes

cell segmntaion 

## Approaches

Simplest : class classification for entire image, then  cell instance segmentation and just use the same predictions for each cell instance.

Training using this apporach for full level label
https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109

then do cell segementation using [this](https://www.kaggle.com/kool777/human-protein-atlas-cell-segmentation-eda)
https://github.com/CellProfiling/HPA-Cell-Segmentation

how would we train first part ? 
straight muliclass classifcaion   on whole image or just target?



### ideas
- how do we get the model to focus attention only on green ?
- rby image then query with green for labels ?
-  use activation map [kaggle post](https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/217395)


### Multi class mAP

precission and recal for each class  not each sample
https://fangdahan.medium.com/calculate-mean-average-precision-map-for-multi-label-classification-b082679d31be