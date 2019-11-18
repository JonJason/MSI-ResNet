"""General training parameters that define the maximum number of
   training epochs, the batch size, and learning rate for the ADAM
   optimization method. To reproduce the results from the paper,
   these values should not be changed.
"""

PARAMS = {
    "n_epochs": 5,
    "batch_size": 10,
    "learning_rate": 1e-5,
    "loss_fn": "kld",
    "metrics": ["kld", "cc", "nss", "auc_borji"]
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
        "input_size": (240, 320)
    },
    "mit1003": {
        "n_train": 1003,
        "val_portion": 0.2,
        "input_size": (360, 360)
    },
    "cat2000": {
        "n_train": 2000,
        "val_portion": 0.2,
        "input_size": (216, 384),
        "categorical": True,
        "n_category": 20
    },
    "cu288": {
        "n_train": 288,
        "val_portion": 0.4,
        "input_size": (360, 360),
        "categorical": True,
        "n_category": 16
    }
}
# "m" for map and "p" for points
MET_SPECS = {
    "kld": "m",
    "cc": "m",
    "kl_cc": "m",
    "nss": "p",
    "auc_borji": "p"
}

THREAD_LIMIT = 8