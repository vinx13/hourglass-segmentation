import h5py
import numpy as np
from scipy.io import loadmat
import xml.etree.cElementTree as ET

prefix = 'VOCdevkit/VOC2010'
f = h5py.File('voc.h5', 'w')
annot = {k:[] for k in {'bbox','name', 'tag'}}
mask_group = f.create_group('mask')
mask_count = 0
mask_index = {}

def get_sample_list(set_name):
    # set_name: 'val', 'train' or 'trainval'
    gt = open(prefix+'/ImageSets/Main/person_' + set_name + '.txt','r')
    samples = []
    for line in  gt.readlines():
        imgname, label = line.split()
        if label == '1':
            samples.append(imgname)
    gt.close()
    return samples

def convert(set_name, tag):
    samples = get_sample_list(set_name)
    convert_mat(samples)
    #print annot['mask']
    for imgname in samples:
        tree = ET.ElementTree(file=prefix + '/Annotations/' + imgname + '.xml')
        root = tree.getroot()
        for obj in tree.iterfind('object'):
            if obj.find('name').text != 'person':
                continue
            bbox = obj.find('bndbox')
            x1,y1,x2,y2 = (bbox.find('xmin').text, bbox.find('ymin').text, bbox.find('xmax').text, bbox.find('ymax').text)
            bbox = [int(x1),int(y1),int(x2),int(y2)]
            annot['bbox'].append(bbox)
            imgname_ = np.zeros(16)
            for i in range(len(imgname)):
                imgname_[i] = ord(imgname[i])
            annot['name'].append(imgname_)
            annot['tag'].append(tag)

def convert_mat(names):
    imgs = []
    for name in names:
        print "Processing " + name
        mat = loadmat('Annotations_Part/' + name)
        objects = mat['anno']['objects'][0][0]
        img = np.zeros_like(objects['mask'][0][0])
        #print img.shape
        for i in range(objects.shape[1]):
            if objects['class_ind'][0][i] != 15: # person
                continue
            parts = objects['parts'][0][i]
            for j in range(parts.shape[1]):
                part_name, mask = parts[0][j]
                img[mask > 0] = j
        mask_group[name] = img


if __name__ == '__main__':
    convert('train',0)
    convert('val',1)
    for k in annot.keys():
        f[k] = np.array(annot[k])
