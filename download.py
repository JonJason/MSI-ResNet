import os
import urllib.request
import zipfile

import requests

def download_salicon(data_path):
    """Downloads the SALICON dataset. Three folders are then created that
       contain the stimuli, binary fixation maps, and blurred saliency
       distributions respectively.

    Args:
        data_path (str): Defines the path where the dataset will be
                         downloaded and extracted to.

    .. seealso:: The code for downloading files from google drive is based
                 on the solution provided at [https://bit.ly/2JSVgMQ].
    """

    print(">> Downloading SALICON dataset...", end="", flush=True)

    default_path = data_path + "salicon/"
    fixations_path = default_path + "fixations/"
    saliency_path = default_path + "maps/"

    os.makedirs(fixations_path, exist_ok=True)
    os.makedirs(saliency_path, exist_ok=True)

    ids = ["1g8j-hTT-51IG1UFwP0xTGhLdgIUCW5e5",
           "0B2hsWbciDVedS1lBZHprdXFoZkU",
           "0B2hsWbciDVedNWJZMlRxeW1PY1U"]

    urls = ["https://drive.google.com/uc?id=" +
            i + "&export=download" for i in ids]

    save_paths = [default_path, fixations_path, saliency_path]

    session = requests.Session()

    for count, url in enumerate(urls):
        response = session.get(url, params={"id": id}, stream=True)
        token = _get_confirm_token(response)

        if token:
            params = {"id": id, "confirm": token}
            response = session.get(url, params=params, stream=True)

        _save_response_content(response, data_path + "tmp.zip")

        with zipfile.ZipFile(data_path + "tmp.zip", "r") as zip_ref:
            for file in zip_ref.namelist():
                if "test" not in file:
                    zip_ref.extract(file, save_paths[count])

    os.rename(default_path + "images", default_path + "stimuli")

    os.remove(data_path + "tmp.zip")

    print("done!", flush=True)


def download_mit1003(data_path):
    """Downloads the MIT1003 dataset. Three folders are then created that
       contain the stimuli, binary fixation maps, and blurred saliency
       distributions respectively.

    Args:
        data_path (str): Defines the path where the dataset will be
                         downloaded and extracted to.
    """

    print(">> Downloading MIT1003 dataset...", end="", flush=True)

    default_path = data_path + "mit1003/"
    stimuli_path = default_path + "stimuli/"
    fixations_path = default_path + "fixations/"
    saliency_path = default_path + "maps/"

    os.makedirs(stimuli_path, exist_ok=True)
    os.makedirs(fixations_path, exist_ok=True)
    os.makedirs(saliency_path, exist_ok=True)

    url = "https://people.csail.mit.edu/tjudd/WherePeopleLook/ALLSTIMULI.zip"
    urllib.request.urlretrieve(url, data_path + "tmp.zip")

    with zipfile.ZipFile(data_path + "tmp.zip", "r") as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith(".jpeg"):
                file_name = os.path.split(file)[1]
                file_path = stimuli_path + file_name

                with open(file_path, "wb") as stimulus:
                    stimulus.write(zip_ref.read(file))

    url = "https://people.csail.mit.edu/tjudd/WherePeopleLook/ALLFIXATIONMAPS.zip"
    urllib.request.urlretrieve(url, data_path + "tmp.zip")

    with zipfile.ZipFile(data_path + "tmp.zip", "r") as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith("Pts.jpg"):
                file_name = os.path.split(file)[1]

                # this file is mistakenly included in the dataset and can be ignored
                if file_name == "i05june05_static_street_boston_p1010764fixPts.jpg":
                    continue

                file_name = file_name.replace("_fixPts", "")
                
                with open(fixations_path + file_name, "wb") as fixations:
                    fixations.write(zip_ref.read(file))

            elif file.endswith("Map.jpg"):
                file_name = os.path.split(file)[1]

                file_name = file_name.replace("_fixMap", "")
                with open(saliency_path + file_name, "wb") as saliency:
                    saliency.write(zip_ref.read(file))

    os.remove(data_path + "tmp.zip")

    print("done!", flush=True)


def download_cat2000(data_path):
    """Downloads the CAT2000 dataset. Three folders are then created that
       contain the stimuli, binary fixation maps, and blurred saliency
       distributions respectively.

    Args:
        data_path (str): Defines the path where the dataset will be
                         downloaded and extracted to.
    """

    print(">> Downloading CAT2000 dataset...", end="", flush=True)

    default_path = data_path + "cat2000/"

    os.makedirs(data_path, exist_ok=True)

    url = "http://saliency.mit.edu/trainSet.zip"
    urllib.request.urlretrieve(url, data_path + "tmp.zip")

    with zipfile.ZipFile(data_path + "tmp.zip", "r") as zip_ref:
        for file in zip_ref.namelist():
            if not("Output" in file or "allFixData" in file):
                zip_ref.extract(file, data_path)

    os.rename(data_path + "trainSet/", default_path)

    os.rename(default_path + "Stimuli", default_path + "stimuli")
    os.rename(default_path + "FIXATIONLOCS", default_path + "fixations")
    os.rename(default_path + "FIXATIONMAPS", default_path + "maps")

    os.remove(data_path + "tmp.zip")

    print("done!", flush=True)
    
def download_dataset(ds_name, parent_path):
    downloader = globals().get("download_" + ds_name, None)
    if downloader is not None:
        downloader(parent_path)
    else:
        raise ValueError('Downloader for dataset "%s" or please specift ' % ds_name +
                        'the path where the images are stored when running the application.' +
                        '\n\nPlease run the command below for help:\n\n' +
                        '\tpython main.py -h')

# TODO: update to your model
def download_pretrained_weights(dest_path, encoder, ds_name, loss_fn_name):
    """Downloads the pre-trained weights for the VGG16 model when
       training or the MSI-Net when testing on new data instances.

    Args:
        dest_path (str): Defines the path where the weights will be
                         downloaded and extracted to.
        key (str): Describes the type of model for which the weights will
                   be downloaded. This contains the device and dataset.

    .. seealso:: The code for downloading files from google drive is based
                 on the solution provided at [https://bit.ly/2JSVgMQ].
    """

    print(">> Downloading pre-trained weights with encoder %s on %s by loss function %s..." % (encoder, ds_name, loss_fn_name), end="", flush=True)

    os.makedirs(dest_path, exist_ok=True)

    ids = {
        "atrous_resnet": {
            "salicon": {
                "kld":"1u5lQXjS9JY5vLFnzItpw6mOlbAWx5p2p",
                "kld_cc": "15S4AY9D87GPfs5jj2x260CNGuXBZ76-M"
            },
            "mit1003": {
                "kld":"150g8MsiAlYEW338LD0hK2OikrW-FWbrn"
            },
            "cat2000":{
            },
            "cu288": {
            }
        },
        "ml_atrous_vgg": {
            "salicon": {
                "kld": "1n34GmSLUxoGduSezYMRO3M2PcJ_jw1gb",
                "kld_cc": "1BvzrEwh68iKadMo5dCrhMahHMA3QeYHY"
            },
            "mit3000": {
                "kld": "1LxBc1x4Dv-Fjpdo88w5ZO7Yn_vMMQeGw"
            }
        }
    }

    url = "https://drive.google.com/uc?id=" + ids[encoder][ds_name][loss_fn_name] + "&export=download"

    session = requests.Session()

    response = session.get(url, params={"id": id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(url, params=params, stream=True)

    _save_response_content(response, dest_path + "tmp.zip")

    with zipfile.ZipFile(dest_path + "tmp.zip", "r") as zip_ref:
        for file in zip_ref.namelist():
            zip_ref.extract(file, dest_path)
    
    os.remove(dest_path + "tmp.zip")

    print("done!", flush=True)


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def _save_response_content(response, file_path):
    chunk_size = 32768

    with open(file_path, "wb") as data:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                data.write(chunk)
