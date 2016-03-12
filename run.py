import subprocess

cmd = 'Rscript sample_submission_test.R'
subprocess.call(cmd, shell = True)

print('#------------------------------------------------#\nRunning Fourier segmentation of images...\n#------------------------------------------------#') 
cmd = 'python segment.py .'
subprocess.call(cmd, shell = True)

print('#------------------------------------------------#\nPreprocessing images for LeNet pass-in...\n#------------------------------------------------#') 
cmd = 'python Preprocessing.py'
subprocess.call(cmd, shell = True)

print('#------------------------------------------------#\nTraining deep learning network...\n#------------------------------------------------#') 
cmd = 'python Train.py Runner --workers=2'
subprocess.call(cmd, shell = True)
