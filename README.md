# KDSB_2015
Polhooper's submission to the Kaggle 2015 Data Science bowl. What we've done here is primarily to integrate the Fourier-based analysis of pixel greyscale oscillations with the [MXNET](https://github.com/dmlc/mxnet/tree/master/example/kaggle-ndsb2) open source deep learner, which implements an LeNet relating grayscale images to diastolic and systolic blood volumes. One of the primariy innovations that we introduce here is to use the masks from the Fourier analysis to identify the region around the left ventricle with a high degree of precision, thereby improving the quality of the training images. 

We didn't spend nearly enough time on this competition to become competitive, but we learned a ton and came up with a at least one solid innovation. Another idea that we haven't presented here is to create generate different CDF functions according to age/gender buckets. 

Enjoy!

**Step 1:** git clone the repo 

**Step 2:** [download the competition data](https://www.kaggle.com/c/second-annual-data-science-bowl) from Kaggle.com, unzip them, and create a `/data` folder within the repo top level directory.   

**Step 3:** Create symlinks of competition data files and directories into `/KDSB_2015/data` top level, e.g.: 
* `ln -s ~/downloads/validate validate`
* `ln -s ~/downloads/train train`
* `ln -s ~/downloads/sample_submission_validate.csv sample_submission_validate.csv`

**Step 4:** `$ python run.py` 
