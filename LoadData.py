import numpy as np
import pickle

folder_path = '/Users/kalryoma/Downloads/cifar-10-batches-py/'

def uncompress(file):
    file_path = folder_path+file
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='bytes')
    labels = np.array(data[b'labels'])
    raw = np.array(data[b'data'], dtype='float')/255.0
    img_data = raw.reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1])
    return img_data, labels

def getLabelName():
    file_path = folder_path+'batches.meta'
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='bytes')[b'label_names']
    return [name.decode() for name in data]

def loadData(bTrain=True):
    if not bTrain:
        images, labels = uncompress('test_batch')
    else:
        images = np.zeros(shape=[50000, 32, 32, 3], dtype=float)
        labels = np.zeros(shape=[50000], dtype=int)
        for i in range(5):
            batch_data, batch_labels = uncompress('data_batch_'+str(i+1))
            start = i*10000
            end = (i+1)*10000
            images[start:end, :] = batch_data
            labels[start:end] = batch_labels
    return images, labels, np.eye(10, dtype=float)[labels]
