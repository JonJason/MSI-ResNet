import tensorflow as tf
import numpy as np
from tensorflow.keras import backend

if backend.image_data_format() == 'channels_last':
    _ch_axis = 3
else:
    _ch_axis = 1

def kld(y_true, y_pred, eps=1e-7):
    y_true /= eps + tf.reduce_sum(y_true, axis=(1,2,3), keepdims=True)
    y_pred /= eps + tf.reduce_sum(y_pred, axis=(1,2,3), keepdims=True)
    loss = y_true * tf.math.log(eps + y_true / (eps + y_pred))
    return tf.reduce_mean(tf.reduce_sum(loss, axis=(1,2,3)))

def nss(y_fixs_true, y_pred):
    y_pred *= 255
    y_fixs_true *= 255
    y_pred = y_pred - tf.reduce_mean(y_pred, axis=(1, 2, 3), keepdims=True)
    y_pred = tf.map_fn(_normalize, y_pred)
    return tf.reduce_mean(y_pred * y_fixs_true)


def cc(y_true, y_pred):
    y_true = tf.map_fn(_normalize, y_true - tf.reduce_mean(y_true, axis=(1,2,3), keepdims=True))
    y_pred = tf.map_fn(_normalize, y_pred - tf.reduce_mean(y_pred, axis=(1,2,3), keepdims=True))
    stacked = tf.reshape(tf.stack([y_true, y_pred], axis=1), [y_true.shape[0], 2,-1]).numpy()
    return tf.reduce_mean(tf.map_fn(lambda y_i: np.corrcoef(y_i[1], y_i[0])[0][1], stacked))

def auc_borji(y_fixs_true, y_pred, n_splits=100, step=0.1, eps=1e-7):

    # reduce channel shape
    new_shape = tf.gather_nd(tf.shape(y_pred), tf.where(tf.range(4)!=_ch_axis))
    y_pred = tf.reshape(y_pred, new_shape)
    y_fixs_true = tf.reshape(y_fixs_true, new_shape)

    scores = []
    for i in range(tf.shape(y_pred)[0]):
        s = tf.reshape(y_pred[i], [-1])
        f = tf.reshape(y_fixs_true[i], [-1])
        s_th = tf.gather_nd(s, tf.where(f > 0))
        n_fixs = tf.shape(tf.where(s_th > 0))[0]
        n_pxs = tf.shape(s)[0]

        # for each fixation, sample Nsplits values from anywhere on the sal map
        r = tf.random.uniform([n_splits, n_fixs], maxval=n_pxs-1, dtype=tf.int32)
        auc = []
        for idxs in r:
            curfix = tf.gather(s,idxs)
            threshes = tf.reverse(tf.range(0, tf.reduce_max(tf.maximum(s_th,curfix))+step, delta=step), [0])
            tp_over_occurrences = tf.cast(tf.reshape(s_th, [1, -1]) >= tf.reshape(threshes, [-1,1]), tf.int32)
            tp = tf.reduce_sum(tp_over_occurrences, axis=1)/n_fixs
            fp_over_occurrences = tf.cast(tf.reshape(curfix, [1, -1]) >= tf.reshape(threshes, [-1,1]), tf.int32)
            fp = tf.reduce_sum(fp_over_occurrences, axis=1)/n_fixs
            tp = tf.concat([[0],tp,[1]], 0)
            fp = tf.concat([[0],fp,[1]], 0)

            auc.append(tf.numpy_function(np.trapz, [tp,fp], tp.dtype))
        scores.append(tf.reduce_mean(auc))

    return tf.reduce_mean(scores)

def kld_nss_cc(y_true, y_fixs_true, y_pred):
    return 10 * kld(y_true, y_pred) - 2 * cc(y_true, y_pred) - nss(y_fixs_true, y_pred)

def _normalize(x):
    return x / tf.math.reduce_std(x) if tf.reduce_max(x) > 0 else x