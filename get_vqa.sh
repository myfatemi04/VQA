#!/bin/sh

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip

mkdir train
mkdir val
unzip v2_Annotations_Train_mscoco.zip -d train
unzip v2_Annotations_Val_mscoco.zip -d val
unzip v2_Questions_Train_mscoco.zip -d train
unzip v2_Questions_Val_mscoco.zip -d val
unzip train2014.zip
mv train2014 train/images
unzip val2014.zip
mv val2014 val/images
