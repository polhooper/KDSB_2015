"""Training script, this is converted from a ipython notebook.
Converted from python3 with fi.__next__ convention 
"""

import pandas as pd
import csv
import logging
import time
import os
import pickle
import sys

import numpy as np
import mxnet as mx
import luigi

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# In[2]:

def get_lenet():
    """ A lenet style net, takes difference of each frame as input.
    """
    source = mx.sym.Variable("data")
    source = (source - 128) * (1.0/128)
    frames = mx.sym.SliceChannel(source, num_outputs=30)
    diffs = [frames[i+1] - frames[i] for i in range(29)]
    source = mx.sym.Concat(*diffs)
    net = mx.sym.Convolution(source, kernel=(5, 5), num_filter=40)
    net = mx.sym.BatchNorm(net, fix_gamma=True)
    net = mx.sym.Activation(net, act_type="relu")
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2,2), stride=(2,2))
    net = mx.sym.Convolution(net, kernel=(3, 3), num_filter=40)
    net = mx.sym.BatchNorm(net, fix_gamma=True)
    net = mx.sym.Activation(net, act_type="relu")
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2,2), stride=(2,2))
    # first fullc
    flatten = mx.symbol.Flatten(net)
    flatten = mx.symbol.Dropout(flatten)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=600)
    # Name the final layer as softmax so it auto matches the naming of data iterator
    # Otherwise we can also change the provide_data in the data iter
    return mx.symbol.LogisticRegressionOutput(data=fc1, name='softmax')

def CRPS(label, pred):
    """ Custom evaluation metric on CRPS.
    """
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1] - 1):
            if pred[i, j] > pred[i, j + 1]:
                pred[i, j + 1] = pred[i, j]
    return np.sum(np.square(label - pred)) / label.size


# In[3]:

def encode_label(label_data):
    """Run encoding to encode the label into the CDF target.
    """
    systole = label_data[:, 1]
    diastole = label_data[:, 2]
    systole_encode = np.array([
            (x < np.arange(600)) for x in systole
        ], dtype=np.uint8)
    diastole_encode = np.array([
            (x < np.arange(600)) for x in diastole
        ], dtype=np.uint8)
    return systole_encode, diastole_encode

def encode_csv(label_csv, systole_csv, diastole_csv):
    systole_encode, diastole_encode = encode_label(np.loadtxt(label_csv, delimiter=","))
    np.savetxt(systole_csv, systole_encode, delimiter=",", fmt="%g")
    np.savetxt(diastole_csv, diastole_encode, delimiter=",", fmt="%g")

# Write encoded label into the target csv
# We use CSV so that not all data need to sit into memory
# You can also use inmemory numpy array if your machine is large enough
encode_csv("../train-label.csv", "../train-stytole.csv", "../train-diastole.csv")


# # Training the stytole net

class RawData(luigi.ExternalTask):
    fname = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.fname)

class TrainNet(luigi.Task):
	name = luigi.Parameter()

	def requires(self):
		return {'data': None, 'label': None,}

	def output(self):
		pass

	def tune_epoch(self):
		c = pd.datetime(2016, 3, 7, 23,59)
		if pd.datetime.now() <= c:
			e = 25 
		else:
			e = 75
		return e


	def run(self):
		data_validate = mx.io.CSVIter(data_csv="../validate-64x64-data.csv", data_shape=(30, 64, 64), batch_size=1)
		network = get_lenet()
		batch_size = 32
		devs = [mx.cpu(0), mx.cpu(0), mx.cpu(0), mx.cpu(0)] #..distribute to multiple cores
		data_train = mx.io.CSVIter(data_csv=self.input()['data'].path, data_shape=(30, 64, 64),
				label_csv=self.input()['label'].path, label_shape=(600,), batch_size=batch_size)


		print "\n%d epochs\n" % self.tune_epoch()
		model = mx.model.FeedForward(ctx=devs,
				symbol             = network,
				num_epoch          = self.tune_epoch(),
				learning_rate      = 0.001,
				wd                 = 0.00001,
				momentum           = 0.9)

		model.fit(X=data_train, eval_metric = mx.metric.np(CRPS))
		prob = model.predict(data_validate)
		prob_fname = "%s_prob" % self.name
		try:
			np.save(prob_fname, prob)
		except:
			pickle.dump(prob, open(prob_fname + '.p', 'wb'))

		pickle.dump(model, open(self.output().path, 'wb'))


class TrainStytole(TrainNet):
	name = luigi.Parameter(default='stytole')

	def requires(self):
		return {'data': RawData('../train-64x64-data.csv'), 
				'label': RawData('../train-stytole.csv')}

	def output(self):
		return luigi.LocalTarget("stytole_model.p")


class TrainDiastole(TrainNet):
	name = luigi.Parameter(default='diastole')

	def requires(self):
		return {'data': RawData('../train-64x64-data.csv'), 
				'label': RawData('../train-diastole.csv')}

	def output(self):
		return luigi.LocalTarget("diastole_model.p")


def accumulate_result(validate_lst, prob):
    sum_result = {}
    cnt_result = {}
    size = prob.shape[0]
    fi = csv.reader(open(validate_lst))
    for i in range(size):
        line = fi.next() # Python2: line = fi.next()
        idx = int(line[0])
        if idx not in cnt_result:
            cnt_result[idx] = 0.
            sum_result[idx] = np.zeros((1, prob.shape[1]))
        cnt_result[idx] += 1
        sum_result[idx] += prob[i, :]
    for i in cnt_result.keys():
        sum_result[i][:] /= cnt_result[i]
    return sum_result

def doHist(data):
    h = np.zeros(600)
    for j in np.ceil(data).astype(int):
        h[j:] += 1
    h /= len(data)
    return h

def submission_helper(pred):
    p = np.zeros(600)
    pred.resize(p.shape)
    p[0] = pred[0]
    for j in range(1, 600):
        a = p[j - 1]
        b = pred[j]
        if b < a:
            p[j] = a
        else:
            p[j] = b
    return p

class Runner(luigi.Task):

	def requires(self):
		return {'stytole':TrainStytole(), 'diastole': TrainDiastole()}

	def output(self):
		return luigi.LocalTarget('submission_mxnet_python.csv')

	def run(self):
		stytole_prob = np.load(self.input()['stytole'].path.replace('model','prob')[:-1] + 'npy')
		diastole_prob = np.load(self.input()['diastole'].path.replace('model','prob')[:-1] +'npy')

		systole_result = accumulate_result("../validate-label.csv", stytole_prob)
		diastole_result = accumulate_result("../validate-label.csv", diastole_prob)

		# we have 2 person missing due to frame selection, use udibr's hist result instead
		train_csv = np.genfromtxt("../train-label.csv", delimiter=',')
		hSystole = doHist(train_csv[:, 1])
		hDiastole = doHist(train_csv[:, 2])

		fi = csv.reader(open("../data/sample_submission_validate.csv"))
		f = open(self.output().path, "w")

		fo = csv.writer(f, lineterminator='\n')
		fo.writerow(fi.next())

		def stretch_submission(data):
			data = np.array(data)
			data -= data.min()
			data *= 1/data.max()
			return data.tolist()
        
		for line in fi:
			idx = line[0]
			key, target = idx.split('_')
			key = int(key)
			out = [idx]
			if key in systole_result:
				if target == 'Diastole':
					add_diastole = list(submission_helper(diastole_result[key]))
					out.extend(stretch_submission(add_diastole))
				else:
					add_systole = list(submission_helper(systole_result[key]))
					out.extend(stretch_submission(add_systole))
			else:
				print("Miss: %s" % idx)
				if target == 'Diastole':
					out.extend(hDiastole)
				else:
					out.extend(hSystole)                
			fo.writerow(out)
		f.close()

if __name__ == "__main__":
    luigi.run()
