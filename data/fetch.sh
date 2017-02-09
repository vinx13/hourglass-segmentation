#!/bin/sh
wget http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/trainval.tar.gz
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
tar -xzf trainval.tar.gz Annotations_Part
tar -xzf VOCtrainval_03-May-2010.tar

python convert.py
