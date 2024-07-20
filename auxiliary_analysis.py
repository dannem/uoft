from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd
from scipy import stats
import mne, os, glob, time, sys, itertools
from sklearn.preprocessing import scale
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from IPython.display import clear_output
from PIL import Image
from scipy.spatial import distance
import matplotlib.pyplot as plt

def show_conf_info(file_name):
  ''' Given a confusion matrix's file_name, prints accuracy information
    :param: file_name
    '''
  confusion = np.genfromtxt(file_name, delimiter=',')
  print(f'Session {i} analysed:')
  print(f'Accuracy of the upright stimuli is {np.mean(squareform(confusion[0:50,0:50]))*100:.4f}%')
  print(f'Accuracy of the upright unfamiliary stimuli is {np.mean(squareform(confusion[0:25,0:25]))*100:.4f}%')
  print(f'Accuracy of the upright famous stimuli is {np.mean(squareform(confusion[25:50,25:50]))*100:.4f}%')
  print(f'Accuracy of the inverted stimuli is {np.mean(squareform(confusion[50:,50:]))*100:.4f}%')
  print(f'Accuracy of the inverted unfamiliar stimuli is {np.mean(squareform(confusion[50:75,50:75]))*100:.4f}%')
  print(f'Accuracy of the inverted famous stimuli is {np.mean(squareform(confusion[75:,75:]))*100:.4f}%')
  print(f'Accuracy of the entire set is {np.mean(squareform(confusion))*100:.4f}%')

  # Correlation results
  corr=np.corrcoef(squareform(confusion[0:25,0:25]),squareform(confusion[50:75,50:75]))
  print(f'Correlation between identity discrimination in upright and inverted famous faces is {corr[0,1]:.3f}')
  corr=np.corrcoef(squareform(confusion[25:50,25:50]),squareform(confusion[75:100,75:100]))
  print(f'Correlation between identity discrimination in upright and inverted unfamiliar faces is {corr[0,1]:.3f}')

def find_folder():
    infolder = '/Volumes/T7/UofT/Data'
    outfolder = './'
    return (infolder, outfolder)

def zscore_threshold(nd_array, dim, threshold, time = False):
    ''' Returns a matrix with zscored and thresholded values along 
        the specified directions.
    :param: nd_array - data
            dim - dimensions to z-transform over as list or int
            threshold - threshold value
    :return: max - zscored and thresholded array with the original nd_array dimensions
    '''
    old_dims = nd_array.shape
    if (time):
        nd_array = nd_array.reshape([old_dims[0], old_dims[1], -1, 5], order='F')
    if isinstance(dim,list):
        dim = tuple(dim)
    mat = stats.zscore(nd_array, axis=dim)
    mat[mat>3] = 3
    mat[mat<-3] = -3
    return mat.reshape([old_dims[0], old_dims[1], old_dims[2]], order='F')

def normalize_epochs(epochs, threshold = None, normalizer = stats.zscore):
  ''' Z-scores within time X els, replaces threshshold (if provided)
    :param epochs: epochs structure
    :param threshold: the threshold absolute z-value. Everything above is threshold, 
                        under is -threshold. Default - False
    :param normalizer: EEG image normalizer. arg1 = data, arg2 = axis.
    :return epochs: - MNE epochs structure 
  '''

  assert (threshold is None) or (threshold >= 0), "threshold should be non-negative"
  assert (type(epochs) in [mne.epochs.EpochsFIF, mne.epochs.EpochsArray]), "epochs should be MNE.epochs structure"

  data = epochs.get_data()
  original_shape = data.shape
  data = data.reshape(-1, data.shape[-1])
  data = normalizer(data, axis = 1)

  if threshold is not None:
    data[np.abs(data) > threshold] = threshold * np.sign(data[np.abs(data) > threshold])
  
  data = data.reshape(original_shape)
  normalized_epochs = mne.EpochsArray(data, epochs.info, events=epochs.events,
                                 event_id=epochs.event_id, tmin=epochs.tmin)
  return normalized_epochs

def block_average(epochs, num_trials, random_pick = False):
    ''' Z-scores within time X els, replaces threshshold (if provided) and averages 
        trials within blocks. 
    :param epochs: epochs structure
    :param num_trials: number of trials to average
    :param num_blocks: number of blocks in the data
    :param random_pick: enables randomly picking the trials
    :return epochs: - MNE epochs structure 
    '''
    
    # Create a new EpochsArray to store the averaged epochs
    averaged_data = []
    averaged_events = []
    # Iterate over conditions
    for condition in epochs.event_id.keys():
        condition_epochs = epochs[condition]
        observations = len(condition_epochs.get_data())
        if (observations % num_trials == 0):
            # Reshape to group epochs into sets of num_trials
            reshaped_data = condition_epochs.get_data().reshape(-1, num_trials, *condition_epochs.get_data().shape[1:])
            reshaped_events = condition_epochs.events.reshape(-1, num_trials, condition_epochs.events.shape[1])
            # Calculate the average across each group of num_trials
            averaged_condition_data = np.mean(reshaped_data, axis=1)
            averaged_condition_events = reshaped_events[:, 0, :] # Keep only the first event of each group
            # Append to the new data and events lists
            averaged_data.append(averaged_condition_data)
            averaged_events.append(averaged_condition_events)
        else:
            # Append full blocks
            number_of_full_blocks = observations // num_trials
            last_block_size = observations - number_of_full_blocks*num_trials

            reshaped_data = condition_epochs.get_data()[:-last_block_size].reshape(-1, num_trials, *condition_epochs.get_data()[:-last_block_size].shape[1:])
            reshaped_events = condition_epochs.events[:-last_block_size].reshape(-1, num_trials, condition_epochs.events[:-last_block_size].shape[1])

            averaged_condition_data = np.mean(reshaped_data, axis=1)
            averaged_condition_events = reshaped_events[:, 0, :]

            averaged_data.append(averaged_condition_data)
            averaged_events.append(averaged_condition_events)

            # Append last block
            reshaped_data = condition_epochs.get_data()[-last_block_size:].reshape(-1, last_block_size, *condition_epochs.get_data()[-last_block_size:].shape[1:])
            reshaped_events = condition_epochs.events[-last_block_size:].reshape(-1, last_block_size, condition_epochs.events[-last_block_size:].shape[1])

            averaged_condition_data = np.mean(reshaped_data, axis=1)
            averaged_condition_events = reshaped_events[:, 0, :]

            averaged_data.append(averaged_condition_data)
            averaged_events.append(averaged_condition_events)

    # Combine the averaged data and create a new EpochsArray
    averaged_data = np.concatenate(averaged_data, axis=0)
    averaged_events = np.concatenate(averaged_events, axis=0)
    # Ensure events are sorted by time
    sort_order = np.argsort(averaged_events[:, 0])
    averaged_data = averaged_data[sort_order]
    averaged_events = averaged_events[sort_order]
    info = epochs.info  # Copy metadata (channel names, sampling rate, etc.)
    averaged_epochs = mne.EpochsArray(averaged_data, info, events=averaged_events,
                                    event_id=epochs.event_id, tmin=epochs.tmin)
    return averaged_epochs

def preprocessing_eeg(fname, event_ids, events=None, segment_times=(-0.1,1), 
                      filt=(0.1,40), crop=None, resample = False):
    ''' Preprocesses EEG data and return segmented epoch structure and events matrix 
        and indices for eeg channels.
        Performs: montage, filter, segmentation, baseline correction, events extraction.
        Optional: cropping, resampling
        
    Args
    :param fname: address + file name
    :param event_ids: the triggers around which to segemnt
    
    Optional args: 
    :param events: customary event matrix (3 columns)
    :param segment_times: customary time points for segmentation (default: -0.1 to 1)
    :param filt: customary filter values (default: Butterworth with 0.1, 
                    40 Hz bandpass and order 4)
    :param crop: cropping times in seconds
    :param resample: desired sampling rate
    
    Returns:
    :return epochs: MNE structure with individual epochs
    '''

    raw = mne.io.read_raw_bdf(fname,
                              preload=True).filter(filt[0], filt[1], method='iir')
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)
    
    # Notch filter if needed
    if filt[1]>60:
        raw.notch_filter(np.arange(60, filt[1], 60), fir_design='firwin')
        
    
    # Cropping
    if crop is not None:
        raw.crop(crop[0], crop[1])
    
    # Segmenting
    if events is None:
        events = mne.find_events(raw, initial_event=True, 
                                 consecutive=True, shortest_event=1)
        
    baseline = (None, 0)  # means from the first instant to t = 0     
    epochs = mne.Epochs(raw, events, event_ids, segment_times[0], segment_times[1],
                    baseline=baseline, preload=True)
    
    # Resampling
    if resample:
        epochs = epochs.resample(resample)
        
    return epochs

def run_eeg_svm(X, Y, cv, aver_n_trials=False, n_pca=False, fft=False):
    ''' Running SVM  
    :param X: 2D data
    :param Y: targets (length should be as 0th dimension of X)
    :param aver_n_trials: number of trials to average over. Default=False
                            no average
    :param n_pca: number of principle component to retain
    :return confusion: confusibility matrix for all available classes
    :return: duration: time that it took to run the procedure
    
    >>> confusion,duration=run_eeg_svm(X,Y,cv=10,aver_n_trials=4,n_pca=100)
    '''
    
    t=time.time()
    dataIn = scale(X)
    if n_pca:
        pca = PCA(n_components = n_pca)
        dataIn = pca.fit_transform(dataIn)
    # else:
    #     print('No PCA')
        
    if fft:
        dataIn = np.abs(np.fft.fft(dataIn))
    # else:
    #     print('No fft')
        
    labelsIn = Y
    results = list()
    #clf = svm.SVC(decision_function_shape='ovo', verbose=0)
    clf = LinearSVC(C=1.0)
    nums = len(list(itertools.combinations(np.unique(labelsIn), 2)))
    for idx, i in enumerate(itertools.combinations(np.unique(labelsIn), 2)):
        # print(i)
        X = dataIn[np.logical_or(labelsIn==i[0], labelsIn==i[1]),:]
        Y = labelsIn[np.logical_or(labelsIn==i[0], labelsIn==i[1])]
        if aver_n_trials:
            X,Y=aver_trials_2D_mat(X,Y,aver_n_trials)
        scores = cross_val_score(clf, X, Y, cv=cv, scoring='accuracy')
        results.append(scores.mean())
        # update_progress(idx / nums)
    
    # update_progress(1)
    confusion = squareform(np.array(results))
    duration=time.time() - t
    duration=duration/60
    # print(f'The overall accuracy is {np.mean(squareform(confusion))*100:.1f} and' +
    #       f' the duration is {duration:.1f} minutes')
    return (confusion, duration)

def convert_epochs_to_2D_array(epochs, times=(0.05,0.65), dims=False, threshold=3,pick_ch=[]):
    ''' Extracts data from the MNE epoch structure, zscores, thresholds and 
        converts into 2D array.
    :param epochs: MNE epochs structure
    :prams times: the begining and the end time within the epoch to extract (default: 50 to 650)
    :param dims: list of dimensions acorss which to z-transform. Default - None.
    :param threshold: threshold value. Default - None
    :return max: - zscored and thresholded array with the original nd_array dimensions
    
    >>> X, Y = convert_epochs_to_2D_array(epochs_02, times=(0.05,0.65), dims=[1, 2])
    '''
    
    picks = mne.pick_types(epochs.info, eeg=True)
    Y = epochs.events[:, 2]
    pick_ch = mne.pick_channels(ch_names=epochs.info['ch_names'], include=pick_ch)
    X = epochs.copy().crop(times[0], times[1]).get_data()[:,pick_ch,:]
    if dims:
        X = zscore_threshold(X, dims, threshold)
    X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]), order='F')
    return (X,Y)

def update_progress(progress):
    ''' Creates a text progress bar and prints out iteration identity
    :param: progress - total number of iterations
    '''
    
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(bar_length * progress))
    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)


def aver_trials_2D_mat(X,Y,n_trials):
    '''Averages n individual trials and updates the traget vector
    :param X: 2D numpy array
    :param Y: 1D target vector
    :param n_trials: number of trials (rows) to average over
    
    >>> X,Y=aver_trials(X,Y,3)
    '''
    
    assert np.unique(Y).shape[0]==2
    class_1,class_2 = np.unique(Y)
    
    
    def comp_mat(X,Y,class_val, n_trials):
        X = X[Y==class_val, :]
        Y = Y[Y==class_val]
        if len(X)%n_trials:
            # padding the matrix if not enough trials
            X = np.pad(X.astype(float), ((0,n_trials-len(X)%n_trials), (0,0)), 
                      'constant', constant_values=(np.nan,))
        X=np.reshape(X,(n_trials,int(len(Y)/n_trials),-1), order='F')
        return np.squeeze(np.nanmean(X, axis=0))
    X_1 = comp_mat(X,Y,class_1, n_trials)
    Y_1 = np.ones((X_1.shape[0],), dtype = 'int' )*class_1
    X_2 = comp_mat(X,Y,class_2, n_trials)
    Y_2 = np.ones((X_2.shape[0],), dtype = 'int' )*class_2
    return np.concatenate((X_1, X_2)),np.concatenate((Y_1, Y_2))

def create_meta_events_perception(evs):
    '''Creates metadata for events
    :param evs: third column from the event array
    :return meta_events: meta events pandas data frame
    
    >>> meta_events=create_meta_events(epochs.events[:,2])
    '''

    meta_events=pd.DataFrame({'ids': evs.astype('int')})
    meta_events['block'] = np.nan
    meta_events['orientation'] = np.nan
    meta_events['origin'] = np.nan
    
    # creating dictionary with triggers
    trigs=list(range(101, 151))+list(range(201, 251))
    dicts={str(x): [1, 1] for x in trigs}
    
    meta_events['orientation']=np.where(meta_events['ids']>200,'in','up')
    east_inds=np.concatenate((np.arange(101,126),np.arange(201,226)))
    meta_events['origin']=np.where(meta_events['ids'].isin(east_inds),'east','west')
    for index, row in meta_events.iterrows():
        # block number
        val=str(int(row['ids']))
        if dicts[val][0] < 5:
            meta_events.loc[index, ['block']] = int(dicts[val][1])
            dicts[val][0] += 1
        else:
            dicts[val][0] = 2
            dicts[val][1] += 1
            meta_events.loc[index, ['block']] = int(dicts[val][1])
    return meta_events


def convert_to_pandas(epochs, num_trials, num_blocks, items_n, average=False, 
                      zscore=False, threshold=False):
    ''' Converts an epcoh array to pandas with times and electrodes as features,
        Optonally Z-scores within time X els, replaces threshshold (if provided) 
        and averages trials within blocks. 
    :param epochs: epochs structure
    :param num_trials: number of trials to average
    :param num_blocks: number of blochs in the data
    :param items_n: number of unique triggers
    :param average: Average in blocks. Default - False
    :param zscore: to zscore along time X electrodes, Default - False
    :param threshold: the threshold absolute z-value. Everything above is threshold, 
                        under is -threshold
    :return df: - Pandas Data Frame 
    
    >>> df = convert_to_pandas(epochs, 4, 11,
                                average = False, zscore = True, threshold = 3)
    '''
    
    df = epochs.to_data_frame()
    df = df.unstack(level = -1)
    df['block']=np.tile(np.arange(1,num_blocks+1).repeat(num_trials),items_n)
    df.reset_index(inplace=True)
    df['condition']=df['condition'].apply(pd.to_numeric)
    east=list(np.arange(101,126))+list(np.arange(201,226))
    if zscore:
        arr = df.iloc[:,2:-1].values
        arr = stats.zscore(arr,axis=1)
        if threshold:
            arr[np.where(arr>threshold)]=threshold
            arr[np.where(arr<-threshold)]=-threshold
        df.iloc[:,2:-1]=arr
    if average:
        df=df.groupby(['block','condition']).mean()
        df.reset_index(inplace=True)
    df['orientation']=np.where(df['condition']>200,'inv','up')
    df['origin']=np.where(df['condition'].isin(east),'east','west')
    return df

def zscore_threshold_epochs(epochs, dims, threshold, time = False):
    ''' Returns a matrix with zscored and thresholded values along 
        the specified directions.
    Args:
    :param epochs: epochs MNE array
    :param dim: dimensions to z-transform over as list or int
    :param threshold: threshold value
    Return:
    :return mat: - zscored and thresholded epochs mne structure
    
    >>> epochs_02 = zscore_threshold_epochs(epochs_02, dims=2, threshold=3) 
    '''
    try:
        temp = epochs.metadata
    except: 
        print('No metadata is found')
        
    data = epochs.get_data()
    data = zscore_threshold(data, dims, threshold, time = time)
    epochs = mne.EpochsArray(data, epochs.info, events=epochs.events, tmin = epochs.tmin)
    epochs.metadata = temp
        
    # Initialize an info structure
    return epochs

def image_to_df(folder, extension = 'tif'):
    '''Loads all images in the folder, removes backgrounds and returns a 
         pandas data frame with column 'names' as file names
    Args: 
    :param folder: the folder with images
    :param extension: image extension. Default - tif
    '''

    cwd = os.getcwd()
    # definig the folder
    os.chdir(folder)
    # loading images
    image_list = []
    flList=list(glob.glob('*.'+extension))
    flList.sort()
    for filename in flList: 
        print(filename)
        print(len(image_list))
        im = Image.open(filename)
        image_list.append(np.array(im))
    data=[]
    [data.append(np.array(np.asarray(i)).ravel()) for i in image_list]
    data=np.array(data)
    mask=np.sum(data,0)==0
    data=data[:,np.logical_not(mask)]
    feature_num=data.shape[1]
    df=pd.DataFrame(data)
    df['names']=flList
    df.names = df.names.str[:-6]
    os.chdir(cwd)
    return df

def theoretical_observer_confusibility_compute(df, feature_num = -1):
    '''Computes theoretical observer confusibility matrix
    Args:
    :param df: data frame with pixels of each identity as rows
    :param feature_num: number of features where pixels are stored 
    starting from the first column
    :return conf_mat: confusibility matrix
    '''

    data=np.array(df.iloc[:,:feature_num])
    conf_mat=np.zeros([data.shape[0],data.shape[0]])
    all_combos = list(itertools.combinations(range(df.shape[0]), 2))
    for i in range(len(all_combos)):
        conf_mat[i]=distance.euclidean(data[i[0],:], data[i[1],:])
        conf_mat[i[1],i[0]]=distance.euclidean(data[i[1],:], data[i[0],:])
    conf_mat=conf_mat/max(conf_mat.ravel())
    return conf_mat

def run_svm_coef(X, Y, aver_n_trials=False, n_pca=False, fft=False):
    ''' Fitting SVM and returning coefficients.
    :param X: 2D data
    :param Y: targets (length should be as 0th dimension of X)
    :param aver_n_trials: number of trials to average over. Default=False
                            no average
    :param n_pca: number of principle component to retain
    :return results: matrix (classification pairs X 64)
    :return: duration: time that it took to run the procedure
    
    >>> results,duration=run_eeg_svm(X,Y,cv=10,aver_n_trials=4,n_pca=100)
    '''
    
    t=time.time()
    dataIn = scale(X)
    if n_pca:
        pca = PCA(n_components = n_pca)
        dataIn = pca.fit_transform(dataIn)
    else:
        print('No PCA')
        
    if fft:
        dataIn = np.abs(np.fft.fft(dataIn))
    else:
        print('No fft')
        
    labelsIn = Y
    results = list()
    #clf = svm.SVC(decision_function_shape='ovo', verbose=0)
    clf = LinearSVC(C=1.0)
    nums = len(list(itertools.combinations(np.unique(labelsIn), 2)))
    for idx, i in enumerate(itertools.combinations(np.unique(labelsIn), 2)):
        print(i)
        X = dataIn[np.logical_or(labelsIn==i[0], labelsIn==i[1]),:]
        Y = labelsIn[np.logical_or(labelsIn==i[0], labelsIn==i[1])]
        if aver_n_trials:
            X,Y=aver_trials_2D_mat(X,Y,aver_n_trials)
        clf.fit(X, Y)
        temp=clf.coef_
        temp=np.reshape(temp,(64,-1),order='F')
        temp=np.mean(temp, axis=1)
        results.append(temp)
        update_progress(idx / nums)
    
    update_progress(1)
    duration=time.time() - t
    duration=duration/60
    print(f' The duration is {duration:.1f} minutes')
    return (results, duration)

def plot_coef_by_cond(epochs_perc, times=(0.05,0.65)):
    # features for perception map
    
    X, Y = convert_epochs_to_2D_array(epochs_perc[[str(x) for x in range(101,126)]], times=times)
    coef, duration = run_svm_coef(X,Y)
    coefs_1=np.mean(np.abs(np.array(coef)),0)
    X, Y = convert_epochs_to_2D_array(epochs_perc[[str(x) for x in range(126,151)]], times=times)
    coef, duration = run_svm_coef(X,Y)
    coefs_2=np.mean(np.abs(np.array(coef)),0)
    X, Y = convert_epochs_to_2D_array(epochs_perc[[str(x) for x in range(201,226)]], times=times)
    coef, duration = run_svm_coef(X,Y)
    coefs_3=np.mean(np.abs(np.array(coef)),0)
    X, Y = convert_epochs_to_2D_array(epochs_perc[[str(x) for x in range(226,251)]], times=times)
    coef, duration = run_svm_coef(X,Y)
    coefs_4=np.mean(np.abs(np.array(coef)),0)
    fig, ax = plt.subplots(figsize=(7.5, 10),nrows=2,ncols=2)
    mne.viz.plot_topomap(coefs_1,epochs_perc.info,vmin=np.min(coefs_1),axes=ax[0,0], show=False)
    ax[0,0].set_title('upright unfamiliar', fontsize=14)
    mne.viz.plot_topomap(coefs_2,epochs_perc.info,vmin=np.min(coefs_2),axes=ax[0,1], show=False)
    ax[0,1].set_title('upright famous', fontsize=14)
    mne.viz.plot_topomap(coefs_3,epochs_perc.info,vmin=np.min(coefs_3),axes=ax[1,0], show=False)
    ax[1,0].set_title('inverted unfamiliar', fontsize=14)
    mne.viz.plot_topomap(coefs_4,epochs_perc.info,vmin=np.min(coefs_4),axes=ax[1,1], show=False)
    ax[1,1].set_title('inverted famous', fontsize=14)
    plt.show()
    return [coefs_1, coefs_2, coefs_3, coefs_4]
    
def convert_lbl(listIn):
    ''' Converts labels to 4 (1-upright unfamiliar, 2-upright famous, 
        3 - inverted unfamiliar, 4 - inverted famous)
    :param: numpy 1D array with labels
    :return: numpy 1D array with labels
    '''
    up_un=np.arange(101,126)
    up_fa=np.arange(126,151)
    in_un=np.arange(201,226)
    in_fa=np.arange(226,251)
    list_out=[]
    for i in listIn:
        if i in up_un:
            list_out.append(1)
        elif i in up_fa:
            list_out.append(2)
        elif i in in_un:
            list_out.append(3)
        elif i in in_fa:
            list_out.append(4)
        else:
            error
    return list_out

def load_multiple_conf_mats(folder, perc_file, imag_file):
    perc_list=[]
    imag_list=[]
    for i,j in zip(perc_file,imag_file):
        perc_list.append(pd.read_csv(i,header=None))
        imag_list.append(pd.read_csv(j, index_col = 0))
    return(perc_list,imag_list)

def inter_sub_corr(df_perc, df_imag):
    
    im_names = ['mcy','sgo','sjo','est','tsw']
    corr_names = ['est','mcy','sjo','sgo','tsw']
    pr_names = ['adi','ani','ama','ago','aza','ekl','evu','epo','eiv','ech','ian','jpi','kda',
               'kgo','mbo','mbe','ofa','pan','pga','rga','siv','tar','tka','yst','ype','ase',
               'aha','ake','cmo','eol','epa','ecl','ero','est','ewa','jla','jal','kpe','kkn',
                'kst','mcy','ndo','npo','owi','pcr','rmc','rwi','sjo','sgo','tsw']
    
    # Theoretical observer:
    #folder1 = 'C:/Users/danne/Documents/UofT/FamousRecon/set3/east'
    #folder2 = 'C:/Users/danne/Documents/UofT/FamousRecon/set3/west'
    folder1 = 'C:\\Users\\nemrodov\\Documents\\Ilya_study\\builder_exp\\set3\\east'
    folder2 = 'C:\\Users\\nemrodov\\Documents\\Ilya_study\\builder_exp\\set3\\west'
    df = pd.concat((image_to_df(folder1),image_to_df(folder2)))
    df['origin'] = ['east']*25+['west']*25
    conf = squareform(pdist(df.iloc[:,:16428],'euclidean'))
    df_to = pd.DataFrame(conf, columns = pr_names, index = pr_names)

    df_perc.columns = [pr_names*2]
    df_perc.index = [pr_names*2]
    idx = pd.IndexSlice
    df_perc_imag = df_perc.loc[idx[im_names], idx[im_names]]
    df_imag = df_imag.reindex(corr_names)
    df_imag = df_imag.transpose().reindex(corr_names).transpose()
    df_to_imag = df_to.loc[idx[im_names], idx[im_names]]
    df_to_imag = df_to_imag.reindex(corr_names)
    df_to_imag = df_to_imag.transpose().reindex(corr_names).transpose()
  
    # compute correlations
    up_corr = np.corrcoef(squareform(df_imag),squareform(df_perc_imag.iloc[:5,:5].values))
    print(f'Correlation between upright perceived and imagery discrimination is {up_corr[1,0]:.2f}')
    in_corr = np.corrcoef(squareform(df_imag),squareform(df_perc_imag.iloc[-5:,-5:].values))
    print(f'Correlation between inverted perceived and imagery discriminations is {in_corr[1,0]:.2f}')
    to_imag_corr = np.corrcoef(squareform(df_to_imag),squareform(df_imag.iloc[:5,:5].values))
    print(f'Correlation between imagery and TO {to_imag_corr[1,0]:.2f}')


    up_perc_to_corr = np.corrcoef(squareform(df_to.values),squareform(df_perc.iloc[:50,:50].values))
    print(f'Correlation between upright perceived and TO discriminations is {up_perc_to_corr[1,0]:.2f}')
    in_perc_to_corr = np.corrcoef(squareform(df_to.values),squareform(df_perc.iloc[50:100,50:100].values))
    print(f'Correlation between inverted perceived and TO discriminations is {in_perc_to_corr[1,0]:.2f}')
    up_perc_to_corr_unf = np.corrcoef(squareform(df_to.values[:25,:25]),squareform(df_perc.iloc[:25,:25].values))
    print(f'Correlation between unfamiliar upright perceived and TO discriminations is {up_perc_to_corr_unf[1,0]:.2f}')
    up_perc_to_corr_fam = np.corrcoef(squareform(df_to.values[25:50,25:50]),squareform(df_perc.iloc[25:50,25:50].values))
    print(f'Correlation between famous upright perceived and TO discriminations is {up_perc_to_corr_fam[1,0]:.2f}')

    in_perc_to_corr_unf = np.corrcoef(squareform(df_to.values[:25,:25]),squareform(df_perc.iloc[50:75,50:75].values))
    print(f'Correlation between unfamiliar inverted perceived and TO discriminations is {in_perc_to_corr_unf[1,0]:.2f}')
    in_perc_to_corr_fam = np.corrcoef(squareform(df_to.values[25:50,25:50]),squareform(df_perc.iloc[75:100,75:100].values))
    print(f'Correlation between famous inverted perceived and TO discriminations is {in_perc_to_corr_fam[1,0]:.2f}')
    return (df_perc_imag, df_imag)

def presenting_results(results, to_print = False, ind = False):
    ''' Aggregates individual results into a panda data frame.
    :param results: list with confusibility matrices
    :return out: - Pandas Data Frame 
    
    >>> df = presenting results(epochs, to_print = True)
    '''

    data = {'up unf' : [np.mean(squareform(i[0:25,0:25]))*100 for i in results],
           'up fam' : [np.mean(squareform(i[25:50,25:50]))*100 for i in results],
           'inv_unf' : [np.mean(squareform(i[50:75,50:75]))*100 for i in results],
           'inv fam' : [np.mean(squareform(i[75:100,75:100]))*100 for i in results],
           'up' : [np.mean(squareform(i[0:50,0:50]))*100 for i in results],
           'inv' : [np.mean(squareform(i[50:100,50:100]))*100 for i in results],
           'all' : [np.mean(squareform(i))*100 for i in results]}
    out = pd.DataFrame.from_dict(data)
    if ind:
        out['Subs']=ind
        out.set_index('Subs')
    if to_print:
        print(out)
    return out

def convert_epochs_to_3D_array(epochs, times=(0.05,0.65), pick_ch=[]):
    ''' Extracts data from the MNE epoch structure, zscores, thresholds and 
        converts into 3D array.
    :param epochs: MNE epochs structure
    :prams times: the begining and the end time within the epoch to extract (default: 50 to 650)
    :return X: - 3D data array
    :return Y: - 1D vector of labels
    
    >>> X, Y = convert_epochs_to_2D_array(epochs_02, times=(0.05,0.65), dims=[1, 2])
    '''

    picks = mne.pick_types(epochs.info, eeg=True)
    Y = epochs.events[:, 2]
    pick_ch = mne.pick_channels(ch_names=epochs.info['ch_names'], include=pick_ch)
    X = epochs.copy().crop(times[0], times[1]).get_data()[:,pick_ch,:]
    return (X,Y)

def isolate_perc_as_imag(df_perc, df_imag):
    
    im_names = ['mcy','sgo','sjo','est','tsw']
    corr_names = ['est','mcy','sjo','sgo','tsw']
    pr_names = ['adi','ani','ama','ago','aza','ekl','evu','epo','eiv','ech','ian','jpi','kda',
               'kgo','mbo','mbe','ofa','pan','pga','rga','siv','tar','tka','yst','ype','ase',
               'aha','ake','cmo','eol','epa','ecl','ero','est','ewa','jla','jal','kpe','kkn',
                'kst','mcy','ndo','npo','owi','pcr','rmc','rwi','sjo','sgo','tsw']
 

    df_perc.columns = [pr_names*2]
    df_perc.index = [pr_names*2]
    idx = pd.IndexSlice
    df_perc_imag = df_perc.loc[idx[im_names], idx[im_names]]
    df_imag = df_imag.reindex(corr_names)
    df_imag = df_imag.transpose().reindex(corr_names).transpose()

    return (df_perc_imag, df_imag)
