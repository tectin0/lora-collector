from math import log
import os
import requests

import modules.scripts as scripts

import logging

logger = logging.getLogger("lora_collector")

base_dir = scripts.basedir()

root_dir = os.sep.join(base_dir.split(os.sep)[:-2])
model_folder_path = os.path.join(root_dir, "models")
lora_folder_path = os.path.join(model_folder_path, "Lora")
lyco_folder_path = os.path.join(model_folder_path, "LyCORIS")


def download_model(url: str, name: str, mtype: str) -> bool:
    """Downloads a model from the civitai website. Only civitai models are supported.

    Args:
        url (str): URL with the model version ID
        name (str): Name of the model
        mtype (str): Type of the model (lora or lyco)
    """
    version_id = url.split("=")[-1]
    download_url = f"http://civitai.com/api/download/models/{version_id}"

    match mtype[:4].lower():
        case "lyco":
            download_folder = lyco_folder_path
        case "lora":
            download_folder = lora_folder_path
        case _:
            logger.warning("Invalid model type. Must be lora or lyco.")
            return False

    r = requests.get(download_url)

    # TODO: .safetensors hardcoded
    with open(f"{download_folder}/{mtype}__{name}.safetensors", "wb") as f:
        f.write(r.content)

    logger.info(f"Downloaded {name} to {download_folder}.")

    return True
