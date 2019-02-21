
from keras.callbacks import TensorBoard

import tensorflow as tf

from LicensePlate import *

import itertools
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras import backend as K
from keras.layers.recurrent import GRU
from keras.optimizers import SGD, Adam

from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, LSTM
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model, load_model


sess = tf.Session()
K.set_session(sess)


def compute_loss(args):
    y_pred, labels, input_length, label_length = args

    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)



def train(img_w, img_h, load):


    count_filters = 18
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 256

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)

    batch_size = 32
    downsample_factor = pool_size ** 2
    train_data = LicensePlateImages('F:/tablice/train3', img_w, img_h, batch_size, downsample_factor)
    train_data.load_data()
    val_data = LicensePlateImages('F:/tablice/valid5', img_w, img_h, batch_size, downsample_factor)
    val_data.load_data()



    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(count_filters, kernel_size, padding='same',
                   activation='relu', kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(count_filters, kernel_size, padding='same',
                   activation='relu', kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * count_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)


    inner = Dense(time_dense_size, activation='relu', name='dense1')(inner)
    # inner = Dropout(0.5)(inner)

    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(
        inner)
    gru1_merged = add([gru_1, gru_1b])
    # inner = Dropout(0.5)(inner)
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
        gru1_merged)

    # inner = Dropout(0.5)(inner)
    inner = Dense(train_data.size_of_chars_set(), kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))

    y_pred = Activation('softmax', name='softmax')(inner)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='labels', shape=[train_data.max_text_len], dtype='float32')
    input_length = Input(name='in_length', shape=[1], dtype='int64')
    label_length = Input(name='lab_length', shape=[1], dtype='int64')

    loss_out = Lambda(compute_loss, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True, update_freq=1)

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    if load:
        model = load_model('F:/tablice/nowe_tabliceee.h5', compile=False)
    else:
        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)


    model.compile(loss='mae', optimizer=adam, metrics=['mae', 'mse' ])

    if not load:

        model.fit_generator(generator=train_data.images_next_batch(),
                            steps_per_epoch=250,
                            epochs=64,
                            validation_data=val_data.images_next_batch(),
                            validation_steps=val_data.n ,
                            verbose=1,
                            callbacks=[tensor_board])

    model.save('F:/tablice/nowe_tabliceee.h5')

    return model

def show_decode_predictinos(out):
    ret = []
    for j in range(out.shape[0]):

        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(chars_set):
                outstr += chars_set[c]
        ret.append(outstr)
    return ret



def per_char_acc(labels, y_pred):
    cnt = 0
    for i, c  in enumerate(labels):
        if labels[i] == y_pred[i]:
            cnt = cnt + 1
    preditc = cnt/7

    return preditc


def detect_number(test_data):

    input_of_net = model.get_layer(name='the_input').input
    out_of_net = model.get_layer(name='softmax').output

    for enter_img, _ in test_data.images_next_batch():

        batchSize = enter_img['the_input'].shape[0]
        X_data = enter_img['the_input']

        net_out_value = sess.run(out_of_net, feed_dict={input_of_net: X_data})

        pred_texts = show_decode_predictinos(net_out_value)
        labels = enter_img['labels']
        texts = []
        for label in labels:
            text = ''.join(list(map(lambda x: chars_set[int(x)], label)))
            texts.append(text)

        counter = 0
        for i in range(batchSize):
            fig = plt.figure(figsize=(10, 10))
            outer = gridspec.GridSpec(2, 1, wspace=10, hspace=0.1)
            ax1 = plt.Subplot(fig, outer[0])
            fig.add_subplot(ax1)

            if pred_texts[i] == texts[i]:
                counter = counter + 1
                plt.text(1, 60, 'Rzeczywisty numer ' + pred_texts[i],
                         bbox=dict(facecolor='green', alpha=0.5), fontsize=14)

            else:
                plt.text(1, 60, 'Rzeczywisty numer ' + pred_texts[i],
                         bbox=dict(facecolor='red', alpha=0.5), fontsize=14)
            img = X_data[i][:, :, 0].T
            ax1.set_xlabel('rozpoznano ' + "{0:.2f}".format(
                100 * per_char_acc(pred_texts[i], texts[i])) + " % literek \n " + 'Licznik: ' + str(
                i + 1) + ' / ' + str(batchSize) + ' \n' + 'rozpoznano ' + "{0:.2f}".format(
                100 * counter / batchSize) + ' % wszystkich tablic')
            ax1.imshow(img, cmap='gray')
            plt.show()

        print('rozpoznano ' + "{0:.2f}".format(100 * counter / batchSize) + ' % wszystkich tablic')
        break




model = train(128, 64,  load=True)

test_data = LicensePlateImages('F:/tablice/test', 128, 64, 38, 1)
test_data.load_data()

detect_number(test_data)






