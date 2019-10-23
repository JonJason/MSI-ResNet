from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os, sys
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

    if args.phase == "train":
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
def train_step(images, ground_truths, model, loss_fn, train_loss, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_fn(ground_truths, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)

@tf.function
def val_step(images, ground_truths, model, loss_fn, val_loss):
    predictions = model(images)
    t_loss = loss_fn(ground_truths, predictions)

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
        else:
            raise FileNotFoundError("Please train model on SALICON database first")
        del salicon_weights
    
    model.summary()

    n_epochs = config.PARAMS["n_epochs"]

    # Preparing progbar
    train_progbar = Progbar(n_train, stateful_metrics=["loss"])
    val_progbar = Progbar(n_val, stateful_metrics=["val_loss"])

    # Preparing 
    loss_fn = tf.keras.losses.KLDivergence()
    optimizer = tf.keras.optimizers.Adam(config.PARAMS["learning_rate"])

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    val_loss = tf.keras.metrics.Mean(name="val_loss")

    ckpts_path = paths["ckpts"] + "%s/%s/" % (encoder, ds_name)
    ckpt = tf.train.Checkpoint(net=model, val_loss=val_loss, optimizer = optimizer)
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
    for epoch in range(start_epoch, n_epochs):

        train_progbar.verbose = 0
        train_progbar.update(0, [("loss", train_loss.result())])
        train_progbar.verbose = 1
        for train_images, train_ground_truths, train_ori_sizes, train_filenames in train_ds:
            train_step(train_images, train_ground_truths, model, loss_fn, train_loss, optimizer)
            train_progbar.add(train_images.shape[0], [("loss", train_loss.result())])

        val_progbar.verbose = 0
        val_progbar.update(0, [("val_loss", val_loss.result())])
        val_progbar.verbose = 1
        for val_images, val_ground_truths, val_ori_sizes, val_filenames in val_ds:
            val_step(val_images, val_ground_truths, model, loss_fn, val_loss)
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

    # Saving model's weights
    print(">> Saving model's weights")
    model.save_weights(paths["weights"] + w_filename_template % (encoder, ds_name))

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

    weights = paths["weights"] + w_filename_template % (encoder, ds_name)
    if os.path.exists(weights):
        model.load_weights(weights)
    else:
        raise FileNotFoundError("Please train model on %s database first" % ds_name.upper())
    del weights

    # Preparing progbar
    test_progbar = Progbar(n_test)


    print(">> Start predicting using model trained on %s..." % ds_name.upper())
    predictions = None
    filenames = None
    ori_sizes = None
    for test_images, test_ori_sizes, test_filenames in test_ds:
        if predictions is None:
            predictions = test_step(test_images, model)
            filenames = test_filenames.numpy()
            ori_sizes = test_ori_sizes
        else:
            tf.concat([predictions, test_step(test_images, model)], axis=0)
            tf.concat([filenames, test_filenames.numpy()], axis=0)
            tf.concat([ori_sizes, test_ori_sizes], axis=0)
        test_progbar.add(test_images.shape[0])
    print("Saving Images")
    results_path = paths["results"]
    for pred, filename, ori_size in zip(predictions, filenames, ori_sizes):
        img = data.postprocess_saliency_map(pred, ori_size)
        tf.io.write_file(results_path + filename.decode("utf-8"), img)

def main():
    """The main function reads the command line arguments, invokes the
       creation of appropriate path variables, and starts the training
       or testing procedure for a model.
    """

    current_path = os.path.dirname(os.path.realpath(__file__))
    default_data_path = current_path + "/data"

    phases_list = ["train", "test"]
    datasets_list = ["salicon", "mit1003", "cat2000", "custom"]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("phase", metavar="PHASE", choices=phases_list,
                        help="sets the network phase (allowed: train or test)")

    parser.add_argument("-d", "--data", metavar="DATA",
                        choices=datasets_list, default=datasets_list[0],
                        help="define which dataset will be used for training \
                              or which trained model is used for testing")

    parser.add_argument("-p", "--path", metavar="DATA_PATH", default=default_data_path,
                        help="specify the path where training data will be \
                              downloaded to or test data is stored")

    parser.add_argument("-c", "--categorical", action="store_true",
                        help="specify wether the data is categorical or not")

    args = parser.parse_args()

    paths = define_paths(current_path, args)

    encoder_name = config.PARAMS["encoder"]

    if args.phase == "train":
        train_model(args.data, paths, encoder_name)
    elif args.phase == "test":
        test_model(args.data, paths, encoder_name, args.categorical)


if __name__ == "__main__":
    main()
