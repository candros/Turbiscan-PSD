import pandas as pd
from sklearn import metrics
import pickle
import os, shutil
from sklearn.model_selection import train_test_split
import scipy
from sklearn.ensemble import ExtraTreesRegressor

def clear_cache():
    '''
    If the cache directory exists, delete all files contained within
    '''
    folder = 'cache'
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

def read_turbiscan(path):
    '''
    Grab all files from the specified directory, if any are excel files write them to the cache directory as .csv files. If they are .csv files, copy them to cache.
    '''

    # If the cache directory has been deleted, create it again
    if not os.path.exists('cache'):
        os.mkdir('cache')
        
    directory = os.listdir(path)
    name = 'excel'
    num = 1
    for fid in directory:
        old_fid = os.path.join(path, fid)
        
        if os.path.splitext(old_fid)[-1] == '.xlsx':
            df = pd.read_excel(old_fid)
            new_name = name + str(num) + '.csv'
            new_fid = os.path.join('cache', new_name)
            num += 1
            df.to_csv(new_fid, index = False)

        elif os.path.splitext(old_fid)[-1] == '.csv':
            new_fid = os.path.join('cache', fid)
            os.path.copy(old_fid, new_fid)

def get_trans_back():
    '''
    Collects all files from the cache directory and parse out the transmission and backscattering data.
    '''
    def process_cols(col):
        '''
        Helper function to separate the time-stamped entries
        '''
        times = []
        names = []
        
        for ind in col:
            items = ind.split()
            names.append(items[0])
            hms = items[-1].split(':')
            
            time = 0 # time is given in minutes
            
            for entry in range(len(hms)):
                if 'd' in hms[entry]:
                    days = int(hms[entry][:-1])
                    time += days * 60 * 24
                elif 'h' in hms[entry]:
                    hours = int(hms[entry][:-1])
                    time += hours * 60
                elif 'm' in hms[entry]:
                    mins = int(hms[entry][:-1])
                    time += mins
                elif 's' in hms[entry]:
                    seconds = int(hms[entry][:-1])
                    time += round(seconds/60)
        
            times.append(time)
        
        df = pd.DataFrame({'sampleid':names, 'minutes':times})
        
        return df

    folder = 'cache'
    list_dir = os.listdir(folder)
    data = [pd.read_csv(os.path.join(folder,x)) for x in list_dir]
    
    transmit = 1
    back = 2
    trans_data = [df.columns[transmit+3*i] for df in data for i in range(int(df.shape[1]/3))]
    back_data = [df.columns[back+3*i] for df in data for i in range(int(df.shape[1]/3))]
    combined_data = pd.concat(data, axis=1)
    combined_data.fillna(value = 0)
    combined_data = combined_data.T
    
    trans_df = combined_data.loc[trans_data, :].reset_index()
    trans_df = pd.concat([process_cols(trans_df['index']), trans_df], axis=1)
    trans_df.drop('index', axis=1, inplace = True)
    
    back_df = combined_data.loc[back_data, :].reset_index()
    back_df = pd.concat([process_cols(back_df['index']), back_df], axis=1)
    back_df.drop('index', axis=1, inplace = True)
    
    results = (back_df, trans_df)

    return results

def calc_auc(data):
    '''
    Calculate the area under the curve (AUC) of each spectra vector
    '''
    auc_x = [int(x) for x in data.columns[2:].values]
    spectra_auc = data.apply(lambda x: metrics.auc(auc_x,x[2:]), axis=1)
    spectra_auc.name = 'auc'
    new_df = pd.concat([data[['sampleid','minutes']], spectra_auc],axis=1)
    new_df = new_df.pivot(index = 'sampleid', columns = 'minutes', values = 'auc')

    return new_df


def calc_mean_trans(data):
    '''
    Calculate the mean value of each spectra vector, method taken from Sun et al 2019
    '''
    mean_trans_x = [int(x) for x in data.columns[1:].values]
    spectra_mean = data.apply(lambda x: np.mean(x[1:].dropna()), axis=1)
    spectra_mean.name = 'mean transmittance'
    new_df = pd.concat([data['minutes'], spectra_mean],axis=1)
    new_df = new_df.pivot(columns = 'minutes', values = 'mean transmittance')
        
    return new_df

def interpolate_spectra(row, nrows = 2100):
    '''
    Interpolate the turbiscan backscattering or transmission spectra so all vectors are a common length
    '''
    row = row.dropna()
    new_x = np.linspace(0, len(row), nrows)
    old_x = np.linspace(0, len(row), len(row))
    new_row = np.interp(new_x, old_x, row.values)
    return new_row

def turbiscan_resampling(prefix, data, X_data, y_data, iters = 300, separate = False):
    '''
    Create "iters" number of train/test/validate splits and fit an extra trees model (n_estimators=500) to each resample
    prefix labels the outputted pickle file
    data is the concatenated dataset
    X_data is the time 0, 2m, 4m, 6m, 8m, 10m backscattering and transmission AUC values
    y_data is a vector of the clay component
    separate is an optional parameter that specifies if your splitting will be random (False) or using the soil samples to split
    
    Output is a pickle file that contains the MSE values of all 300 random splits and the information for the highest-performing random split, structured as follows: 
    output = (list of 300 MSE values, (final X train, final X test, final y train, final y test), (X validate, y validate), final clay model, final silt model, final sand model)
    '''
    # To address the concern of lack of variance in the triplicates, perform train-test splits on the triplicates
    def create_separate_triplicates(X_in, y_in, split):

        X = X_in.copy()
        y = y_in.copy()
        # Create a new index for locating the unique triplicates
        X['sample'] = X.index
        X['sample'] = X['sample'].apply(lambda x: x[:-1])
        X.set_index('sample', inplace = True, append= True)
        y.index = X.index

        # split the samples by triplicate
        uniques = X.index.get_level_values('sample').unique()
        vector = []
        while len(vector) < int(split*len(uniques)):
            new = random.randint(0, len(uniques)-1)
            if new not in vector:
                vector.append(new)

        # create the train-test split and drop the appended index
        X_test = X.loc[pd.IndexSlice[:,uniques[vector]],:].droplevel('sample')
        X_train = X.drop(X_test.index).droplevel('sample')
        y_test = y.loc[pd.IndexSlice[:,uniques[vector]]].droplevel('sample')
        y_train = y.drop(y_test.index).droplevel('sample')

        return (X_train, X_test, y_train, y_test)

    # Starting MSE for the models to beat
    mse1 = 1000
    mse2 = 1000
    range1 = 50

    # Record of train-test splits stats to keep track of split instability
    runs = {'mse_clay': [],'mse_silt': [],'mse_sand': [], 'r2_clay': [],'r2_silt': [], 'r2_sand': []}

    # Iterate through different train-test splits, find the best split.
    while len(runs['mse_clay']) < iters:

        if separate:
            # Perform the train-test split ensuring that triplicates are in different train, test, validate sets
            X_train1, X_val, y_train1, y_val = create_separate_triplicates(X_data, y_data, split = 0.20)
            X_train, X_test, y_train, y_test = create_separate_triplicates(X_train1, y_train1, split = 0.23)
        else:
            # perform the train-test split across the three different targets
            X_train1, X_val, y_train1, y_val = train_test_split(X_data, y_data, test_size = 0.16)
            X_train, X_test, y_train, y_test = train_test_split(X_train1, y_train1, test_size = 0.20)

        # Check to verify the train-test splits have equal variance (levene's test): homoscedasticity
        (stat, p_val) = scipy.stats.levene(y_train, y_test)

        # if the runs do not have equal variance, do not use that split
        if p_val > 0.05:  
            y_test_clay = data.loc[y_test.index, 'clay']
            y_train_clay = data.loc[y_train.index, 'clay']
            y_test_silt = data.loc[y_test.index, 'silt']
            y_train_silt = data.loc[y_train.index, 'silt']
            y_test_sand = data.loc[y_test.index, 'sand']
            y_train_sand = data.loc[y_train.index, 'sand']

            # fit clay
            clay_model = ExtraTreesRegressor(n_estimators=500)
            clay_model.fit(X_train, y_train_clay)
            clay_true = y_test_clay
            clay_pred = clay_model.predict(X_test)
            clay_mse = metrics.mean_squared_error(clay_true, clay_pred)
            clay_r2 = metrics.r2_score(clay_true, clay_pred)
            runs['mse_clay'].append(clay_mse)
            runs['r2_clay'].append(clay_r2)

            # fit silt
            silt_model = ExtraTreesRegressor(n_estimators = 500)                          
            silt_model.fit(X_train, y_train_silt)
            silt_true = y_test_silt
            silt_pred = silt_model.predict(X_test)
            silt_mse = metrics.mean_squared_error(silt_true, silt_pred)
            silt_r2 = metrics.r2_score(silt_true, silt_pred)
            runs['mse_silt'].append(silt_mse)
            runs['r2_silt'].append(silt_r2)
            silt_diff = abs(silt_pred - silt_true)
            silt_range = max(silt_diff) - min(silt_diff)

            # fit sand
            sand_model = ExtraTreesRegressor(n_estimators = 500)
            sand_model.fit(X_train, y_train_sand)
            sand_true = y_test_sand
            sand_pred = sand_model.predict(X_test)
            sand_mse = metrics.mean_squared_error(sand_true, sand_pred)
            sand_r2 = metrics.r2_score(sand_true, sand_pred)
            runs['mse_sand'].append(sand_mse)
            runs['r2_sand'].append(sand_r2)

            # At every iteration, check to see if the train-test split has improved the model
            if (clay_mse < mse1) and (silt_mse < mse2):

                # If a train-test split with a better metric appears, report all models and metrics associated with it
                final_clay_model = clay_model
                final_clay_mse = clay_mse
                final_clay_r2 = clay_r2
                final_silt_model = silt_model
                final_silt_mse = silt_mse
                final_silt_r2 = silt_r2
                final_sand_model = sand_model
                final_sand_mse = sand_mse
                final_sand_r2 = sand_r2
                X_validate, y_validate = (X_val, y_val)

                final_train = (X_train, X_test, y_train, y_test)
                mse1 = clay_mse
                mse2 = silt_mse
                range1 = silt_range

    # Save the run statistics
    results = (runs, final_train, (X_validate, y_validate), final_clay_model, final_silt_model, final_sand_model)
    with open(prefix+"_OSC_results.pkl", "wb") as fp:
        pickle.dump(results, fp)

        
def open_pickle(fid):
    '''
    Open a pickle file
    '''
    with open(fid, 'rb') as f:
        item = pickle.load(f)
    return item

    