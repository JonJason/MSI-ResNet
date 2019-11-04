from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os
import logging

import pprint

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.utils import Progbar

import config
import data
from model import MyModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

def define_paths(current_path, args):
    """A helper function to define all relevant path elements for the
       locations of data, weights, and the results from either training
       or testing a model.

    Args:
        current_path (str): The absolute path string of this script.
        args (object): A namescpace object with values from command line.

    Returns:
        dict: A dictionary with all path elements.
    """

    if os.path.isfile(args.path):
        data_path = args.path
    else:
        data_path = os.path.join(args.path, "")

    results_path = current_path + "/results/"
    weights_path = current_path + "/weights/"
    ckpts_path = weights_path + "ckpts/"

    if args.action == "train":
        if args.data not in data_path:
            data_path += args.data + "/"

    paths = {
        "data": data_path,
        "results": results_path,
        "weights": weights_path,
        "ckpts": ckpts_path
    }

    return paths

@tf.function
def train_step(images, y_true, model, loss_fn, train_loss, optimizer):
    with tf.GradientTape() as tape:
        y_pred = model(images)
        loss = loss_fn(y_true, y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    return y_pred

@tf.function
def val_step(images, y_true, model, loss_fn, val_loss):
    y_pred = model(images)
    t_loss = loss_fn(y_true, y_pred)

    val_loss(t_loss)

@tf.function
def test_step(images, model):
    return model(images)

def train_model(ds_name, paths, encoder):
    """The main function for executing network training. It loads the specified
       dataset iterator, saliency model, and helper classes. Training is then
       performed in a new session by iterating over all batches for a number of
       epochs. After validation on an independent set, the model is saved and
       the training history is updated.

    Args:
        ds_name (str): Denotes the dataset to be used during training.
        paths (dict, str): A dictionary with all path elements.
    """

    w_filename_template = "/%s_%s_weights.h5" # [encoder]_[ds_name]_weights.h5

    (train_ds, n_train), (val_ds, n_val) = data.load_train_dataset(ds_name, paths["data"])
    
    print(">> Preparing model with encoder %s..." % encoder)

    model = MyModel(encoder, ds_name, "train")

    if ds_name != "salicon":
        salicon_weights = paths["weights"] + w_filename_template % (encoder, "salicon")
        if os.path.exists(salicon_weights):
            model.load_weights(salicon_weights)
            print("Salicon weights are loaded!")
        else:
            raise FileNotFoundError("Please train model on SALICON database first")
        del salicon_weights

    model.summary()

    n_epochs = config.PARAMS["n_epochs"]

    # Preparing 
    loss_fn = kl_divergence
    optimizer = tf.keras.optimizers.Adam(config.PARAMS["learning_rate"])

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    val_loss = tf.keras.metrics.Mean(name="val_loss")

    ckpts_path = paths["ckpts"] + "%s/%s/" % (encoder, ds_name)
    ckpt = tf.train.Checkpoint(net=model, val_loss=val_loss)
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpts_path, max_to_keep=n_epochs,
                                                checkpoint_name="model__val_loss__optimizer__ckpt")
    start_epoch = 0
    
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        val_loss.reset_states()
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        print ('Checkpoint restored:\n{}'.format(ckpt_manager.latest_checkpoint))

    print(">> Start training model on %s..." % ds_name.upper())
    if ds_name == "salicon" and start_epoch < 2:
        model.freeze_unfreeze_encoder_trained_layers(True)
    for epoch in range(start_epoch, n_epochs):
        if ds_name == "salicon" and epoch == 2:
            model.freeze_unfreeze_encoder_trained_layers(False)

        train_progbar = Progbar(n_train, stateful_metrics=["loss"])
        for train_images, train_y_true, train_ori_sizes, train_filenames in train_ds:
            y_pred = train_step(train_images, train_y_true, model, loss_fn, train_loss, optimizer)
            train_progbar.add(train_images.shape[0], [("loss", train_loss.result())])

        val_progbar = Progbar(n_val, stateful_metrics=["val_loss"])
        for val_images, val_y_true, val_ori_sizes, val_filenames in val_ds:
            val_step(val_images, val_y_true, model, loss_fn, val_loss)
            val_progbar.add(val_images.shape[0], [("val_loss", val_loss.result())])

        train_loss_result = train_loss.result()
        val_loss_result = val_loss.result()
        template = 'Epoch {} - Loss: {} - Val Loss: {}'
        print(template.format(epoch+1,
            ('%.4f' if train_loss_result > 1e-3 else '%.4e') % train_loss_result,
            ('%.4f' if val_loss_result > 1e-3 else '%.4e') % val_loss_result
        ))
        
        ckpt_manager.save()

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        val_loss.reset_states()

    # Picking best result
    print(">> Picking best result")
    min_val_loss = None

    for i, checkpoint in enumerate(ckpt_manager.checkpoints):
        ckpt.restore(checkpoint)
        val_loss_result = val_loss.result()
        if min_val_loss is None or min_val_loss > val_loss_result:
            min_val_loss = val_loss_result
            min_index = i

    ckpt.restore(ckpt_manager.checkpoints[min_index])
    print("best result picked -> epoch: %d - val_loss: %s" % (min_index + 1,
        ('%.4f' if min_val_loss > 1e-3 else '%.4e') % min_val_loss))

    # Saving model's weights
    print(">> Saving model's weights")
    dest_path = paths["weights"] + w_filename_template % (encoder, ds_name)
    model.save_weights(dest_path)
    print("weights are saved to:\n%s" % dest_path)

def test_model(ds_name, paths, encoder, categorical=False):
    """The main function for executing network testing. It loads the specified
       dataset iterator and optimized saliency model. By default, when no model
       checkpoint is found locally, the pretrained weights will be downloaded.

    Args:
        ds_name (str): Denotes the dataset that was used during training.
        encoder (str): the name of the encoder want to be used to predict.
        paths (dict, str): A dictionary with all path elements.
    """

    w_filename_template = "/%s_%s_weights.h5" # [encoder]_[ds_name]_weights.h5

    (test_ds, n_test) = data.load_test_dataset(ds_name, paths["data"], categorical)
    
    print(">> Preparing model with encoder %s..." % encoder)

    model = MyModel(encoder, ds_name, "test")

    weights_path = paths["weights"] + w_filename_template % (encoder, ds_name)
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print("Salicon weights are loaded!")
    else:
        raise FileNotFoundError("Please train model on %s database first" % ds_name.upper())
    del weights_path



    print(">> Start predicting using model trained on %s..." % ds_name.upper())
    y_pred = None
    filenames = None
    ori_sizes = None
    # Preparing progbar
    test_progbar = Progbar(n_test)
    for test_images, test_ori_sizes, test_filenames in test_ds:
        pred = test_step(test_images, model)
        if y_pred is None:
            y_pred = pred
            filenames = test_filenames
            ori_sizes = test_ori_sizes
        else:
            y_pred = tf.concat([y_pred, pred], axis=0)
            filenames = tf.concat([filenames, test_filenames], axis=0)
            ori_sizes = tf.concat([ori_sizes, test_ori_sizes], axis=0)
        test_progbar.add(test_images.shape[0])
    print("Saving Images")
    results_path = paths["results"]
    for pred, filename, ori_size in zip(y_pred, filenames.numpy(), ori_sizes):
        img = data.postprocess_saliency_map(pred, ori_size)
        tf.io.write_file(results_path + filename.decode("utf-8"), img)

def kl_divergence(y_true, y_pred, eps=1e-7):
    sum_per_image = tf.reduce_sum(y_true, axis=(1, 2, 3), keepdims=True)
    y_true /= eps + sum_per_image

    sum_per_image = tf.reduce_sum(y_pred, axis=(1, 2, 3), keepdims=True)
    y_pred /= eps + sum_per_image
    loss = y_true * tf.math.log(eps + y_true / (eps + y_pred))
    return tf.reduce_mean(tf.reduce_sum(loss, axis=(1, 2, 3)))

def main():
    """The main function reads the command line arguments, invokes the
       creation of appropriate path variables, and starts the training
       or testing procedure for a model.
    """

    current_path = os.path.dirname(os.path.realpath(__file__))
    default_data_path = current_path + "/data"

    actions_list = ["train", "test", "summary"]
    datasets_list = ["salicon", "mit1003", "cat2000", "custom"]
    encoders_list = ["atrous_resnet", "ml_atrous_vgg"]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("action", metavar="ACTION", choices=actions_list,
                        help="specify the action (allowed: %s)" % " or ".join(actions_list))

    parser.add_argument("-d", "--data", metavar="DATA",
                        choices=datasets_list, default=datasets_list[0],
                        help="define which dataset will be used for training \
                              or which trained model is used for testing")

    parser.add_argument("-p", "--path", metavar="DATA_PATH", default=default_data_path,
                        help="specify the path where training data will be \
                              downloaded to or test data is stored")

    parser.add_argument("-e", "--encoder", metavar="ENCODER",
                        choices=encoders_list, default=encoders_list[0],
                        help="specify the action (available: %s)" % " or ".join(encoders_list))

    parser.add_argument("-c", "--categorical", action="store_true",
                        help="specify wether the data is categorical or not")

    args = parser.parse_args()

    paths = define_paths(current_path, args)

    encoder_name = args.encoder

    if args.action == "train":
        train_model(args.data, paths, encoder_name)
    elif args.action == "test":
        test_model(args.data, paths, encoder_name, args.categorical)
    elif args.action == "summary":
        
        model = MyModel(encoder_name, args.data, "test")
        model.summary()


if __name__ == "__main__":
    main()
