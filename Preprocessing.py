"""Preprocessing script.

This script walks over the directories and dump the frames into a csv file
"""
import os
import csv
import sys
import random
import scipy
import numpy as np
import pandas as pd
import dicom
import utils as ut
import re
import matplotlib
from glob import glob
from skimage import io, transform

def mkdir(fname):
   try:
       os.mkdir(fname)
   except:
       pass

def get_frames(root_path, start, end):    
    subject_dirs = os.walk(root_path).next()[1]
    subject_dirs.sort(key = float)
    start_index = subject_dirs.index(start)
    end_index = subject_dirs.index(end) + 1
    sax_list = []
    mask_list = []
    for subject in subject_dirs[start_index:end_index]:
        subject_path = os.path.join(root_path, subject) 
        time_dirs = os.walk(subject_path).next()[1]
        time_0_path = os.path.join(subject_path, time_dirs[0])
        time_0_files = os.listdir(time_0_path)
        n_slices = len(re.findall('color.png', ' '.join(time_0_files))) 
        for sax in range(n_slices):
            sax_tmp = []
            mask_tmp = []
            for time in time_dirs:
                sax_path = os.path.join(subject_path, time, "slice%02d_color.png" % (sax,))
                mask_path = os.path.join(subject_path, time, "slice%02d_mask.png" % (sax,))
                sax_tmp.append(sax_path)
                mask_tmp.append(mask_path)
            sax_list.append(sax_tmp)
            mask_list.append(mask_tmp)
        print('list parsed for subject: %s' % subject)
    return([sax_list, mask_list])
                
def get_label_map(fname):
   labelmap = {}
   fi = open(fname)
   fi.readline()
   for line in fi:
       arr = line.split(',')
       labelmap[int(arr[0])] = line
   return labelmap


def write_label_csv(fname, frames, label_map):
   fo = open(fname, "w")
   for lst in frames:
       subject_id = lst[0].split('/').index('output') + 1
       index = int(lst[0].split("/")[subject_id])
       if label_map != None:
           fo.write(label_map[index])
       else:
           fo.write("%d,0,0\n" % index)
   fo.close()


def write_data_csv(fname, frames, preprocess_func=None, final_img_size=64):
    """Write data to csv file"""
    fdata = open(fname, "w")
    dwriter = csv.writer(fdata)
    counter = 0
    result = []
    for lst in frames:
        print "subject %s\n" % lst[0].split('/')[2]
        data = []
        
        #..read all mask data into an array:        
        for path in lst:
            arr = scipy.ndimage.imread(path, flatten=True)
                        
            subject_path = '/'.join(path.split('/')[:-1]) 
            subject_files = os.listdir(subject_path)
            mask_save_path = "/".join([subject_path[:-7], 'smooth_mask'])
            try:
                smoothed_mask = np.load(mask_save_path + '.npy')
            except:
                mask_paths = glob("%s/*/*mask.png" % subject_path[:-7])
                masks = [matplotlib.image.imread(x) for x in mask_paths]
                masks = np.array([ut.mask_convert(x) for x in masks])
                mask_thresh = len(masks)
                mask_sums = masks.sum(axis = 0)
                smoothed_mask = scipy.ndimage.filters.gaussian_filter(mask_sums, 3)
                np.save(mask_save_path, smoothed_mask)
            x_idx, y_idx = np.where(smoothed_mask == np.nanmax(smoothed_mask))
            
            def mean_idx(idx):
                return int(idx.mean())
            
            x_idx = mean_idx(x_idx)
            y_idx = mean_idx(y_idx)
            center = (x_idx, y_idx)
            mask = matplotlib.image.imread(path.replace('color', 'mask'))
            norm_arr = arr/arr.max()
            img = preprocess_func(norm_arr, center, final_img_size)
            dst_path = path.rsplit(".", 1)[0] + ".64x64.jpg"
            scipy.misc.imsave(dst_path, img)
            result.append(dst_path)
            data.append(img)
        data = np.array(data, dtype=np.uint8)
        data = data.reshape(data.size)
        dwriter.writerow(data)
        counter += 1
        if counter % 10 == 0:
            print("%d slices processed" % counter)
    print("All finished, %d slices in total" % counter)
    fdata.close()
    return result

def get_boundaries(arr, centers, size):
    """ Adjust centroid_calc centers to image size
    """
    half_size = size/2
    x_center, y_center = centers
    
    x_box = np.array([x_center - half_size, x_center + half_size])
    y_box = np.array([y_center - half_size, y_center + half_size])
        
    def diff_check(box, res):
        if box[1] > res: 
            diff = box[1] - res
            box = box - diff
        elif box[0] < 0: 
            diff = abs(box[0])
            box = box + diff
        return box.tolist()
        
    adj_x_box = diff_check(x_box, arr.shape[0])
    adj_y_box = diff_check(y_box, arr.shape[1])
    return (adj_x_box, adj_y_box)


def crop_resize(img, center, size):
    """crop center and resize"""
    if img.shape[0] < img.shape[1]:
        img = img.T
        center = (center[1], center[0])
    
    x, y = get_boundaries(img, center, size)
    
    cropped_img = img[x[0]:x[1], y[0]:y[1]]
    c = pd.datetime(2016, 3, 7, 23,59)
    if pd.datetime.now() > c:
	    cropped_img = cropped_img/cropped_img.max()
    cropped_img *= 255
    
    return cropped_img.astype("uint8")


def local_split(train_index):
   random.seed(0)
   train_index = set(train_index)
   all_index = sorted(train_index)
   num_test = int(len(all_index) / 5)
   random.shuffle(all_index)
   train_set = set(all_index[num_test:])
   test_set = set(all_index[:num_test])
   return train_set, test_set


def split_csv(src_csv, split_to_train, train_csv, test_csv):
   ftrain = open(train_csv, "w")
   ftest = open(test_csv, "w")
   cnt = 0
   for l in open(src_csv):
       if split_to_train[cnt]:
           ftrain.write(l)
       else:
           ftest.write(l)
       cnt = cnt + 1
   ftrain.close()
   ftest.close()

# Load the list of all the training frames, and shuffle them
# Shuffle the training frames

train_tmp = get_frames("./output", '1', '500')
train_frames = train_tmp[0]
random.seed(10)
random.shuffle(train_frames)
train_masks = train_tmp[1]
random.seed(10)
random.shuffle(train_masks)

validate_tmp = get_frames("./output", '701', '1140')
validate_frames = validate_tmp[0]
validate_masks = validate_tmp[1]

# Write the corresponding label information of each frame into file.
write_label_csv("./train-label.csv", train_frames, get_label_map("./data/train.csv"))
write_label_csv("./validate-label.csv", validate_frames, None)

# Dump the data of each frame into a CSV file, apply crop to 64 preprocessor
train_lst = write_data_csv("./train-64x64-data.csv", train_frames, preprocess_func=crop_resize)
valid_lst = write_data_csv("./validate-64x64-data.csv", validate_frames, preprocess_func=crop_resize)
