"""General training parameters that define the maximum number of
   training epochs, the batch size, and learning rate for the ADAM
   optimization method. To reproduce the results from the paper,
   these values should not be changed.
"""

PARAMS = {
    "n_epochs": 10,
    "batch_size": 10,
    "learning_rate": 1e-5,
    "encoder": "atrous_resnet"
}

"""The predefined input image sizes for each of the 3 datasets.
   To reproduce the results from the paper, these values should
   not be changed. They must be divisible by 8 due to the model's
   downsampling operations. Furthermore, all pretrained models
   for download were trained on these image dimensions.
"""

SPECS = {
    "salicon": {
        "n_train": 10000,
        "n_val": 5000,
        "img_size": (240, 320)
    },
    "mit1003": {
        "n_train": 1003,
        "val_portion": 0.2,
        "img_size": (360, 360),
        "sal_map_suffix": "_fixMap"
    },
    "cat2000": {
        "n_train": 2000,
        "val_portion": 0.2,
        "img_size": (216, 384),
        "categorical": True,
        "n_category": 20
    }
}