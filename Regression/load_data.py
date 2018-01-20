import numpy as np
import pickle

def load_data(nsplit):
    DATA_FILE_NAME = 't_price_filtered.pkl'

    with open(DATA_FILE_NAME, 'rb') as f:
        data = pickle.load(f)
    
    train_data = data[0:48000 * 300].reshape([nsplit, -1, 300])
    test_data = data[48000*300:60000*300].reshape([12000, 300])
    
    train_label = np.zeros([nsplit, train_data.shape[1]])
    test_label = np.zeros(test_data.shape[0])
    
    for isplit in range(nsplit):
        pos = int(300 * ((.6 - .4) * isplit / (nsplit - 1) + .4))
        for idx in range(train_data.shape[1]):
            train_label[isplit][idx] = sorted(train_data[isplit][idx])[pos]
            
    for idx in range(test_data.shape[0]):
        test_label[idx] = sorted(test_data[idx])[int(300 * .6)]
        
    return train_data, train_label, test_data, test_label
