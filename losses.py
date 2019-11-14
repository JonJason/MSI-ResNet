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
    y_pred = _normalize(y_pred)
    return tf.reduce_mean(tf.gather_nd(y_pred, tf.where(y_fixs_true == 1)))

def cc(y_true, y_pred):
    y_true = _normalize(y_true)
    y_pred = _normalize(y_pred)
    stacked = tf.reshape(tf.stack([y_true, y_pred], axis=1), [y_true.shape[0], 2,-1])
    scores = tf.zeros([stacked.shape[0]],dtype=stacked.dtype)
    for i in range(y_pred.shape[0]):
        score = tf.numpy_function(lambda x,y:
            np.corrcoef(x,y)[0][1].astype("float32"), [stacked[i][1], stacked[i][0]], stacked.dtype)
        scores = tf.tensor_scatter_nd_update(scores, [[i]], [score])
    return tf.reduce_mean(scores)

def auc_borji(y_fixs_true, y_pred, n_splits=100, step=0.1, eps=1e-7):

    # reduce channel shape
    new_shape = tf.gather_nd(tf.shape(y_pred), tf.where(tf.range(4)!=_ch_axis))
    y_pred = tf.reshape(y_pred, new_shape)
    y_fixs_true = tf.reshape(y_fixs_true, new_shape)

    scores = tf.zeros([y_pred.shape[0]],dtype=y_pred.dtype)
    for i in range(scores.shape[0]):
        s = tf.reshape(y_pred[i], [-1])
        f = tf.reshape(y_fixs_true[i], [-1])
        s_th = tf.gather_nd(s, tf.where(f > 0))
        n_fixs = s_th.shape[0]
        n_pxs = s.shape[0]

        # for each fixation, sample Nsplits values from anywhere on the sal map
        r = tf.random.uniform([n_splits, n_fixs], maxval=n_pxs-1, dtype=tf.int32)
        
        auc = tf.zeros([n_splits],dtype=y_pred.dtype)
        
        for j in range(n_splits):
            curfix = tf.gather(s, tf.gather(r, j))
            threshes = tf.reverse(tf.range(0, tf.reduce_max(tf.maximum(s_th,curfix))+step, delta=step), [0])

            # rounding threshes as workaround to imprecision of tf.range of float
            _dec_factor = 1/step
            threshes = tf.round(threshes * _dec_factor) / _dec_factor

            # continue
            tp_over_occurrences = tf.cast(tf.reshape(s_th, [1, -1]) >= tf.reshape(threshes, [-1,1]), tf.int32)
            tp = tf.cast(tf.reduce_sum(tp_over_occurrences, axis=1)/n_fixs, tf.float32)
            fp_over_occurrences = tf.cast(tf.reshape(curfix, [1, -1]) >= tf.reshape(threshes, [-1,1]), tf.int32)
            fp = tf.cast(tf.reduce_sum(fp_over_occurrences, axis=1)/n_fixs, tf.float32)
            tp = tf.concat([[0],tp,[1]], 0)
            fp = tf.concat([[0],fp,[1]], 0)
            
            auc = tf.tensor_scatter_nd_update(auc, [[j]], [tf.numpy_function(np.trapz, [tp,fp], tp.dtype)])
        scores = tf.tensor_scatter_nd_update(scores, [[i]], [tf.reduce_mean(auc)])

    return tf.reduce_mean(scores)

def kld_cc(y_true, y_pred):
    kld_score = kld(y_true, y_pred)
    cc_score = cc(y_true, y_pred)
    return kld_score - cc_score + 1

def _normalize(x):
    x = x - tf.reduce_mean(x, axis=(1, 2, 3), keepdims=True)
    return _update_nd(x, tf.reduce_max(x, axis=(1, 2, 3)) > 0,
        lambda _x: _x/tf.math.reduce_std(_x, axis=(1,2,3), keepdims=True))

def _update_nd(x, whereabouts, updater):
    pos = tf.where(whereabouts)
    _x = tf.gather_nd(x, pos)
    new_x = updater(_x)
    return tf.tensor_scatter_nd_update(x, pos, new_x)
