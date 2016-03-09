#..helper functions (see below for classes)
#------------------------------------------
import numpy as np 
import pandas as pd 
import os 
import re 
from sklearn.preprocessing import LabelEncoder
import matplotlib

def mask_convert(inarray):
    ref_pix = inarray[0, 0]
    bin_mat  = inarray == ref_pix
    sum_mat = bin_mat.sum(axis = 2)
    ref_pix_2 = sum_mat[0, 0]
    return sum_mat != ref_pix_2

def centroid_calc(bin_mask):
    bin_mask = mask_convert(inarray)
    row_sums = bin_mask.sum(axis = 1)
    col_sums = bin_mask.sum(axis = 0)
    col_sums = col_sums[::-1]
    
    def base_calc(inarray):
        count = 0 
        for e in inarray: 
            if e == 0: 
                count += 1
            else: 
                break 
        return(count)

    x_base = base_calc(row_sums)
    y_base = base_calc(col_sums)
    
    x_sums = np.array([float(x) for x in row_sums if x != 0])
    x_weights = x_sums/x_sums.sum()
    y_sums = np.array([float(x) for x in col_sums if x != 0])
    y_weights = y_sums/y_sums.sum()
    
    def add_calc(weights):
        pos_vec = np.array(range(len(weights))) + 1
        tmp_vec = pos_vec * weights
        return(int(tmp_vec.sum()))
    
    x_center = x_base + add_calc(x_weights)
    y_center = y_base + add_calc(y_weights)
    return((x_center, y_center))


def progress_print(i, N, inc = 10): 
    k = int(N/10)
    if i % k == 0:
        progress = int(i/k) * 10  
        print('{}% complete'.format(progress))

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def CRPS_row(p, V_m):
    p = np.array(p)
    v = np.array(range(len(p)))
    h = v >= V_m
    sq_dists = (p - h)**2
    return(np.sum(sq_dists)/len(sq_dists)) 

def CRPS_mean(df): 
    """
    Function recieves pandas dataframe as an input with the first column being
    a column of truths (V_m) and the subsequent 600 columns being cumulatively
    summed probabilities
    """
    crps_vec = df.apply(CRPS_row, axis = 1)
    crps_sum = np.sum(crps_vec)
    return(crps_sum/len(crps_vec))

def doHist(data):
    h = np.zeros(600)
    for j in np.ceil(data.values).astype(int):
        h[j:] += 1
    h /= len(data)
    return h

def remove_nas(df):
    for field in df.columns.values: 
        if df[field].dtypes != 'object':
            if any(np.isnan(np.array(df[field]))):
                df = df.drop(field, axis = 1)
    return(df)

def cat_encode(df, onehot = False, to_omit = ['PatientName', 'SOPInstanceUID']):
    cols_to_omit = [x for x in to_omit if x in df.columns.values]
    if len(cols_to_omit) > 0:
        df = df.drop(to_omit, axis = 1)
    
    if not onehot:
        for field in df.columns.values: 
            if df[field].dtype == 'O':
                data = np.array(df[field])
                data = np.unique(np.sort(data))
                le = LabelEncoder()
                le.fit(data)
                df[field] = le.transform(df[field])
        return(df)
    else: 
        return(pd.get_dummies(df))

def match_ref_cdf(cdf_dict, cdf_type, age):
    #..break out list of tuples based on key file 
    idx = []
    for key in cdf_dict.keys():
        tmp = key.split('.')
        cdf_type_tmp = tmp[1]
        age_tmp = tmp[2].split('_')[1].split('-')
        age_tmp = (float(age_tmp[0]), float(age_tmp[1]))
        idx.append(cdf_type_tmp == cdf_type and age >= age_tmp[0] and age < age_tmp[1])
    
    match_key = np.array(cdf_dict.keys())[np.array(idx)]
    
    if not len(match_key) == 1: 
        print('Matching key error in match_ref_cdf')
    else: 
        match_key = str(match_key[0])
    
    return([match_key, cdf_dict[match_key]])

def cdf_dict_scanner(cdfs_dir):
    
    file_names = os.listdir(cdfs_dir)
    regex = re.compile('cdf')
    cdf_files = [x for x in file_names if regex.search(x)]
    
    cdf_dict = {}
    for cdf in cdf_files: 
        cdf_dict[cdf] = np.genfromtxt(os.path.join(cdfs_dir, cdf), delimiter = ',')
    
    return(cdf_dict)

def dem_data_cast(cdfs_dir, cdf_ests_file, meta_file):
    """
    Function to append demographic data to data frame based on PatientID. 
    The first argument, cdfs, should be submission-ready file with ID's 
    associated with systolic and diastolic volumes associated. Features that 
    result from applying this function are: 
    (1) Original meta dile file
    (2) A new binary feature, 'diastolic'
    (3) Difference statistic columns between the input CDFs and reference CDFs
    """
    
    cdf_dict = cdf_dict_scanner(cdfs_dir)
    
    cdf_ests = pd.read_csv(cdf_ests_file)
    meta = pd.read_csv(meta_file)
    
    #..to demographic data file add following fields 
    #..(1) Diastole
    #..(2) CDF type
    #..(3) Difference metrics between the base and reference curves 
    #..(4) Replacement dummy for instances where base CDF replaces estimated CDF 
    
    header = meta.columns.values 
    out_array = np.append(header, ['diastole', 'cdf_type', 'mean_diff', 'mean_abs_diff', 'mean_sq_diff', 'replace_dummy'])
    
    print('casting demographic data...')
    N = cdf_ests.shape[0]
    for i in range(N):
        progress_print(i, N) 
        row = cdf_ests.iloc[i,]
        est_id = row['Id']
        tmp = est_id.split('_')
        ID = int(tmp[0])
        cdf_type = tmp[1]
        age = float(meta.loc[meta['PatientID'] == ID, 'PatientAge'])
        
        out_row = np.transpose(np.array(meta.ix[meta['PatientID'] == ID, ]))[:,0]
        
        ref_cdf_tmp = match_ref_cdf(cdf_dict, cdf_type, age)    
        ref_cdf = ref_cdf_tmp[1]
        est_cdf = cdf_ests.ix[cdf_ests['Id'] == est_id, 1:].as_matrix()[0,:]
        
        if(sum(est_cdf == 1) >= 599):
            est_cdf = ref_cdf 
            replace_dummy = 1
        else: 
            replace_dummy = 0 
        
        diastole = int(cdf_type == 'Diastole')    
        cdf_type = ref_cdf_tmp[0]
        mean_diff = np.mean(est_cdf - ref_cdf)
        mean_abs_diff = np.mean(abs(est_cdf - ref_cdf))
        mean_sq_diff = np.mean((est_cdf - ref_cdf) ** 2)
        
        out_row = np.append(out_row, [diastole, cdf_type, mean_diff, mean_abs_diff, mean_sq_diff, replace_dummy])
        out_array = np.row_stack((out_array, out_row))
    
    out = pd.DataFrame(out_array[1:,:], columns = out_array[0,:])
    return(pd.DataFrame.convert_objects(out, convert_numeric = True))

def optimize_alphas(comp_curves, truths, cdfs_dir, meta_file, increment = 0.02):
    meta = pd.read_csv(meta_file)
    comps = pd.read_csv(comp_curves)
    truths = pd.read_csv(truths)
    truth_dict = {}
    for i in range(truths.shape[0]):
        truth_dict[str(int(truths.iloc[i,0])) + '_Diastole'] = truths.iloc[i, 2]
        truth_dict[str(int(truths.iloc[i,0])) + '_Systole'] = truths.iloc[i, 1]
    
    weights = np.arange(0, 1 + increment, increment)
    alphas = []
    ids = []
    
    cdf_dict = cdf_dict_scanner(cdfs_dir) 
    
    print('alpha optimization....')
    N = comps.shape[0]
    for i in range(N):
        progress_print(i, N)
        row = comps.iloc[i,]
        est_id = row['Id']
        tmp = est_id.split('_')
        ID = int(tmp[0])
        cdf_type = tmp[1]
        age = float(meta.loc[meta['PatientID'] == ID, 'PatientAge'])
        
        cdf = match_ref_cdf(cdf_dict, cdf_type, age)[1]
        
        est = np.array(comps.iloc[i,1:])
        ID = comps.iloc[i,0]
        ids.append(ID)
        V_m = truth_dict[ID]
        
        scores = [CRPS_row(w * est + (1 - w) * cdf, V_m) for w in weights]
        idx = [i for i, j in enumerate(scores) if j == min(scores)]
        alphas.append(float(weights[idx]))
        
        i = pd.Series(ids, name = 'Id')
        a = pd.Series(alphas, name = 'alphas')
    
    return(pd.concat([i, a], axis = 1))

#..Classes 
#---------

class boot_cdf:
    def __init__(self, df, B):
        nrows = df.shape[0]
        sum_systole = np.zeros([1, 600])
        sum_diastole = np.zeros([1, 600])
        print('generating bootstrapped CDF estimates...')
        for b in range(B):
            progress_print(b, B)
            idx = np.random.randint(nrows, size = nrows)
            tmp_data = df.iloc[idx]
            sum_systole = sum_systole + doHist(tmp_data.Systole)
            sum_diastole = sum_diastole + doHist(tmp_data.Diastole)
        
            self.hSystole = sum_systole/B
            self.hDiastole = sum_diastole/B
