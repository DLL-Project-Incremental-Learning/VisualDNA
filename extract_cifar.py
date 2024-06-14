from __future__ import print_function
import numpy as np
import pickle
import cv2
import os

def load_cifar_pickle(path, file):
    with open(os.path.join(path, file), 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
        images = data_dict[b'data']
        images = np.reshape(images, (10000, 3, 32, 32))
        labels = np.array(data_dict[b'labels'])
        print("Loaded {} labelled images from {}.".format(images.shape[0], file))
    return images, labels

def load_cifar_categories(path, file):
    with open(os.path.join(path, file), 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
        return [label.decode('utf-8') for label in data_dict[b'label_names']]

def save_cifar_image(array, path):
    # array is 3x32x32. cv2 needs 32x32x3
    array = array.transpose(1, 2, 0)
    # array is RGB. cv2 needs BGR
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # save to PNG file
    return cv2.imwrite(path, array)

if __name__ == '__main__':
    base_dir = './dataset'
    pickle_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']

    categories = load_cifar_categories(os.path.join(base_dir, 'cifar-10-batches-py'), 'batches.meta')
    print(categories)

    for pickle_file in pickle_files:
        images, labels = load_cifar_pickle(os.path.join(base_dir, 'cifar-10-batches-py'), pickle_file)
        for i in range(len(images)):
            cat = categories[labels[i]]
            out_dir = os.path.join(base_dir, 'cifar10', cat)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            save_cifar_image(images[i], os.path.join(out_dir, '{}_image_{}.png'.format(pickle_file, i)))
