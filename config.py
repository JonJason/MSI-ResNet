"""General training parameters that define the maximum number of
   training epochs, the batch size, and learning rate for the ADAM
   optimization method. To reproduce the results from the paper,
   these values should not be changed.
"""

PARAMS = {
    "n_epochs": 10,
    "batch_size": 10,
    "learning_rate": 1e-5,
    "loss_fn": "kld_nss_cc",
    "metrics": ["kld_nss_cc", "cc", "nss"],
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
    }
}

# loss function input format ("f" for fixs_map, "s" for sal_map, else for both)
METRICS = {
    "kld": "s",
    "nss": "f",
    "cc": "s",
    "auc_borji": "f",
    "kld_nss_cc": "_"
}