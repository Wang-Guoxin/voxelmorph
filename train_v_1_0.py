import os, sys
import voxelmorph as vxm
import neurite as ne
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2.')
import h5py
import matplotlib.pyplot as plt

def normalized_data(result_file_path):
    all_data = []
    with h5py.File(result_file_path, 'r') as result_file:
        for i in range(len(result_file)):
            dataset_name = f'data_{i}'
            if dataset_name in result_file:
                data = result_file[dataset_name][:] # [:] means "get all the data"
                all_data.append(data)

    all_data_array = np.array(all_data)

    min_val = np.min(all_data_array)
    max_val = np.max(all_data_array)
    normalized_data = (all_data_array - min_val) / (max_val - min_val)

    print(f'Shape of all_data_array: {all_data_array.shape}')
    # print(f'Data: {all_data_array[:3]}')
    return normalized_data

def Datalogger(normalized_data):
    # split data into training and validation sets
    x_train = normalized_data
    nb_val = 10
    x_val = x_train[-nb_val:, ...]  # this indexing means "the last nb_val entries" of the zeroth axis
    x_train = x_train[:-nb_val, ...]
    print('shape of x_train and x_val', x_train.shape, x_val.shape)

    # pad images with zeros to increase size by 2
    pad_amount = ((0, 0), (2,2), (2,2))
    x_train = np.pad(x_train, pad_amount, 'constant')
    x_val = np.pad(x_val, pad_amount, 'constant')
    print('shape after pad:', x_train.shape, x_val.shape)
    return x_train, x_val

def vxm_data_generator(x_data, batch_size=1):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    # preliminary sizing
    vol_shape = x_data.shape[1:] # extract data shape
    ndims = len(vol_shape)
    print('vol_shape', vol_shape)
    print('ndims', ndims)
    
    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    
    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = x_data[idx1, ..., np.newaxis]
        idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        fixed_images = x_data[idx2, ..., np.newaxis]
        inputs = [moving_images, fixed_images]
        
        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare 
        # the resulting moved image with the fixed image. 
        # we also wish to penalize the deformation field. 
        outputs = [fixed_images, zero_phi]
        yield (inputs, outputs)

def plot_history(hist, loss_name='loss'):
    # Simple function to plot training history.
    plt.figure()
    plt.plot(hist.epoch, hist.history[loss_name], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

def train(x_train, x_val):
    """
    configure unet input shape (concatenation of moving and fixed images)
    """
    ndim = 2 # number of image dimensions (2D or 3D)
    unet_input_features = 2 # use two input images (moving and fixed images)
    inshape = (*x_train.shape[1:], unet_input_features)
    print('unet input shape:', inshape)

    # configure unet features 
    nb_features = [
        [32, 32, 32, 32],         # encoder features
        [32, 32, 32, 32, 32, 16]  # decoder features
    ]

    # build model using VxmDense
    inshape = x_train.shape[1:]
    vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)
    print('input shape: ', ', '.join([str(t.shape) for t in vxm_model.inputs]))
    print('output shape:', ', '.join([str(t.shape) for t in vxm_model.outputs]))

    # voxelmorph has a variety of custom loss classes
    losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
    lambda_param = 0.05
    loss_weights = [1, lambda_param]
    vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

    # train
    train_generator = vxm_data_generator(x_train)
    in_sample, out_sample = next(train_generator)
    nb_epochs = 10
    steps_per_epoch = 100
    hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2)
    plot_history(hist)

    # visualize
    images = [img[0, :, :, 0] for img in in_sample + out_sample] 
    titles = ['moving', 'fixed', 'moved ground-truth (fixed)', 'zeros']
    ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)
    
    val_generator = vxm_data_generator(x_val, batch_size = 1)
    val_input, _ = next(val_generator)
    val_pred = vxm_model.predict(val_input)
    
    # visualize
    images = [img[0, :, :, 0] for img in val_input + val_pred] 
    titles = ['moving', 'fixed', 'moved', 'flow']
    ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)
    ne.plot.flow([val_pred[1].squeeze()], width=50)
        
if __name__ == "__main__":
    result_file_path = r'C:\Users\39349\Documents\GitHub\TransMorph_Transformer_for_Medical_Image_Registration\result.h5'
    normalized_data_result = normalized_data(result_file_path)
    x_train, x_val = Datalogger(normalized_data_result)
    train(x_train, x_val)