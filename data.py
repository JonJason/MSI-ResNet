import os

import math
import glob

import numpy as np
import tensorflow as tf

import config
import download

from tensorflow.keras import backend

# TODO: fix comments for all functions
def load_train_dataset(ds_name, data_path):
    spec = config.SPECS[ds_name]
    target_size = spec["img_size"]
    sal_map_suffix = spec.get("sal_map_suffix", "")
    category_depth = 1 if spec.get("categorical", False) else 0

    n_train = spec["n_train"]
    has_val = "n_val" in spec
    if not has_val and "val_portion" not in spec:
        raise NotImplementedError('Either "n_val" or "val_portion" must be specified')

    train_imgs_dir = data_path + "stimuli"
    train_maps_dir = data_path + "maps"

    if has_val:
        n_val = spec["n_val"]
        train_imgs_dir += "/train"
        train_maps_dir += "/train"
    else:
        n_val = math.floor(n_train * spec["val_portion"])

    # download dataset if not exists
    if not os.path.exists(data_path):
        parent_path = os.path.dirname(os.path.dirname(data_path))
        parent_path = os.path.join(parent_path, "")

        download.download_dataset(ds_name, parent_path)
    
    # loading dataset
    train_x = _get_file_list(train_imgs_dir, category_depth)
    train_y = _get_file_list(train_maps_dir, category_depth, suffix=sal_map_suffix)
    _check_consistency(zip(train_x, train_y), n_train, suffix=sal_map_suffix)
    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))

    if has_val:
        train_ds = train_ds.shuffle(n_train, reshuffle_each_iteration=False)
        valid_list_x = _get_file_list(data_path + "stimuli/val", category_depth)
        valid_list_y = _get_file_list(data_path + "maps/val", category_depth, suffix=sal_map_suffix)
        _check_consistency(zip(valid_list_x, valid_list_y), n_val, suffix=sal_map_suffix)

        val_ds = tf.data.Dataset.from_tensor_slices((valid_list_x, valid_list_y))
        val_ds = val_ds.shuffle(n_val, reshuffle_each_iteration=False)
    else:
        if category_depth > 0:
            n_category = spec["n_category"]
            n_per_cat = n_train // n_category
            n_val_per_cat = math.floor(n_per_cat * spec["val_portion"])
            for i in range(n_category):
                cat_train_ds = train_ds.take(n_per_cat)
                cat_train_ds = cat_train_ds.shuffle(n_per_cat)
                if i == 0:
                    val_ds = cat_train_ds.take(n_val_per_cat)
                    new_train_ds = cat_train_ds.skip(n_val_per_cat)
                else:
                    val_ds.concatenate(cat_train_ds.take(n_val_per_cat))
                    new_train_ds.concatenate(cat_train_ds.skip(n_val_per_cat))
                train_ds = train_ds.skip(n_per_cat)
            train_ds = new_train_ds
        else:
            train_ds = train_ds.shuffle(n_train, reshuffle_each_iteration=False)
            val_ds = train_ds.take(n_val)
            train_ds = train_ds.skip(n_val)

        n_train -= n_val

    train_ds = _prepare_image_ds(train_ds, target_size, category_depth)
    val_ds = _prepare_image_ds(val_ds, target_size, category_depth)

    return ((train_ds, n_train), (val_ds, n_val))

def load_test_dataset(ds_name, data_path, categorical=False):
    
    spec = config.SPECS[ds_name]
    target_size = spec["img_size"]
    category_depth = 1 if categorical else 0

    test_x = _get_file_list(data_path, category_depth)
    n_test = len(test_x) 
    test_ds = tf.data.Dataset.from_tensor_slices(test_x)
    test_ds = _prepare_image_ds(test_ds, target_size, category_depth)

    return (test_ds, n_test)

def postprocess_saliency_map(saliency_map, target_size):
    """This function resizes and crops a single saliency map to the original
       dimensions of the input image. The output is then encoded as a jpeg
       file suitable for saving to disk.

    Args:
        saliency_map (tensor, float32): 3D tensor that holds the values of a
                                        saliency map in the range from 0 to 1.
        target_size (tensor, int32): 1D tensor that specifies the size to which
                                     the saliency map is resized and cropped.

    Returns:
        tensor, str: A tensor of the saliency map encoded as a jpeg file.
    """

    if backend.image_data_format() == 'channels_first':
        saliency_map = tf.transpose(saliency_map, (1, 2, 0))

    saliency_map = _restore_size(saliency_map, target_size)

    saliency_map = tf.image.convert_image_dtype(saliency_map, tf.uint8)

    saliency_map_jpeg = tf.image.encode_jpeg(saliency_map, "grayscale", quality=100)

    return saliency_map_jpeg

def _prepare_image_ds(ds, target_size, category_depth, one_at_a_time=False):
    """Here the list of file directories is shuffled (only when training),
       loaded, batched, and prefetched to ensure high GPU utilization.

    Args:
        files (list, str): A list that holds the paths to all file instances.
        target_size (tuple, int): A tuple that specifies the size to which
                                  the data will be reshaped.
        shuffle (bool): Determines whether the dataset will be shuffled or not.
        one_at_a_time (bool, optional): Flag that decides whether the batch size must
                                 be 1 or can take any value. Defaults to False.

    Returns:
        object: A dataset object that contains the batched and prefetched data
                instances along with their shapes and file paths.
    """
    ds = ds.map(lambda *io_paths_pair: _parse_inputs(io_paths_pair, target_size, category_depth),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    batch_size = 1 if one_at_a_time else config.PARAMS["batch_size"]

    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=5)
    
    return ds

def _parse_inputs(io_paths_pair, target_size, category_depth):
    """This function reads image data dependent on the image type and
       whether it constitutes a stimulus or saliency map. All instances
       are then reshaped and padded to yield the target dimensionality.

    Args:
        io_paths_pair (tuple, str): A tuple with the paths to all file instances.
                            The first element contains the stimuli and, if
                            present, the second one the ground truth maps.
        target_size (tuple, int): A tuple that specifies the size to which
                                  the data will be reshaped.

    Returns:
        list: A list that holds the image instances along with their
              shapes and file paths.
    """
    
    image_list = []
    for i, filename in enumerate(io_paths_pair):
        image_str = tf.io.read_file(filename)
        channels = 3 if i == 0 else 1 # wether it is the stimuli or the map

        image = tf.cond(tf.image.is_jpeg(image_str),
            lambda: tf.image.decode_jpeg(image_str, channels=channels),
            lambda: tf.image.decode_png(image_str, channels=channels))

        # from int value (0 - 255) to float value (0 - 1)
        image = tf.image.convert_image_dtype(image, tf.float64)

        original_size = tf.shape(image)[:2]

        image = tf.image.resize_with_pad(image, target_size[0], target_size[1], method=tf.image.ResizeMethod.AREA)

        # if it is the stimuli, transform to 0 - 255 to comply with the input formats
        if i == 0:
            image = image * 255
        
        if backend.image_data_format() == 'channels_first':
            image = tf.transpose(image, (2, 0, 1))
        
        image_list.append(image)

    image_list.append(original_size)

    output_filename = tf.strings.regex_replace(io_paths_pair[0],"\\\\","/")
    output_filename = tf.strings.split(output_filename, '/')[-(category_depth + 1):]
    image_list.append(tf.strings.reduce_join(output_filename, axis=0, separator="/"))

    return image_list

def _restore_size(image, target_size):
    current_size = tf.shape(image)[:2]
    height_ratio = target_size[0] / current_size[0]
    width_ratio = target_size[1] / current_size[1]
    
    target_ratio = tf.maximum(height_ratio, width_ratio)
    new_size = tf.cast(current_size, tf.float64) * target_ratio
    new_size = tf.cast(tf.round(new_size), tf.int32)
    
    image = tf.image.resize(image, new_size,
                                    method=tf.image.ResizeMethod.BICUBIC,
                                    preserve_aspect_ratio=False)

    offset = tf.cast((new_size - target_size) /2, tf.int32)
    
    cropped_image = image[offset[0]:offset[0] + target_size[0], offset[1]:offset[1] + target_size[1]]

    return cropped_image


def _get_file_list(data_path, depth=0, suffix=""):
    """This function detects all image files within the specified parent
       directory for either training or testing. The path content cannot
       be empty, otherwise an error occurs.

    Args:
        data_path (str): Points to the directory where training or testing
                         data instances are stored.

    Returns:
        list, str: A sorted list that holds the paths to all file instances.
    """

    data_list = []

    if os.path.isfile(data_path):
        data_list.append(data_path)
    else:
        file_pattern = data_path + ("/*" * (depth+1)) + suffix + ".*"
        data_list = [f for f in glob.glob(file_pattern) if f.endswith(('.jpg', '.jpeg', '.png'))]

    data_list.sort()

    if not data_list:
        raise FileNotFoundError("No data was found")

    return data_list

def _check_consistency(zipped_file_lists, n_total_files, suffix=""):
    """A consistency check that makes sure all files could successfully be
       found and stimuli names correspond to the ones of ground truth maps.

    Args:
        zipped_file_lists (tuple, str): A tuple of train and valid path names.
        n_total_files (int): The total number of files expected in the list.
    """
    assert len(list(zipped_file_lists)) == n_total_files, "Files are missing"

    for file_tuple in zipped_file_lists:
        file_names = [os.path.basename(entry) for entry in list(file_tuple)]
        file_names = [os.path.splitext(entry)[0] for entry in file_names]
        if suffix != "":
            file_names = [entry.replace(suffix, "") for entry in file_names]

        assert len(set(file_names)) == 1, "File name mismatch"