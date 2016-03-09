# KDSB_2015
Polhooper's submission to the Kaggle 2015 Data Science bowl. What we've done here is primarily to integrate the Fourier-based analysis of pixel greyscale oscillations with the [MXNET](https://github.com/dmlc/mxnet/tree/master/example/kaggle-ndsb2) open source deep learner, which implements an LeNet relating grayscale images to diastolic and systolic blood volumes. One of the primariy innovations that we introduce here is to use the masks from the Fourier analysis to identify the region around the left ventricle with a high degree of precision, thereby improving the quality of the training images. 

We didn't spend nearly enough time on this competition to become competitive, but we learned a ton and came up with a at least one solid innovation. Another idea that we haven't presented here is to create generate different CDF functions according to age/gender buckets. 

Some quick notes: 
* Getting mxnet setup can be seriously painful. [Check back in with our blog](blog.polhooper.com) in a few days for guidance on that if you're having trouble with setup. 
* We use [luigi](https://pypi.python.org/pypi/luigi) to parallelize training of the diasoltic and systolic networks on the same machine. [Check out luigi tutorials](http://help.mortardata.com/technologies/luigi/first_luigi_script) for guidance on how to start a luigi server and run python scripts built with luigi in mind. We've tried to make this as hands-free as possible here. 
* Depending on your machine, `segment.py` and `Preprocessing.py` will take over a full day to run together. Be sure to allow some time for your machine to not go to sleep. This is all single-threaded and clumsy right now, but could easily be parallelized. 

Enjoy!

**Step 1:** git clone this repo 

**Step 2:** [download the competition data](https://www.kaggle.com/c/second-annual-data-science-bowl) from Kaggle.com, unzip them, and create a `/data` folder within the repo top level directory.   

**Step 3:** Create symlinks of competition data files and directories into `/KDSB_2015/data` top level, e.g.: 
* `ln -s ~/downloads/validate validate`
* `ln -s ~/downloads/train train`
* `ln -s ~/downloads/sample_submission_validate.csv sample_submission_validate.csv`

**Step 4:** Open a separate terminal window and fire up a luigi server. After running `pip install luigi`, one easy way to do this is to clone Spotify's [luigi GitHub repo](https://github.com/spotify/luigi) and run the `luigid` binary executable within that repo, e.g. (on our machine) `$  ~/documents/repos/luigi/bin/luigid`

**Step 5:** `$ python run.py` 
