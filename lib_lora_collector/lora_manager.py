import json
from logging import config
import os

from lib_lora_collector.models import load_models, Model

import modules.scripts as scripts
from lib_lora_collector.downloader import download_model

from copy import deepcopy

import logging

base_dir = scripts.basedir()
root_dir = os.sep.join(base_dir.split(os.sep)[:-2])

logger = logging.getLogger("lora_collector")


class LoraManager:
    def __init__(self):
        super().__init__()

        self.is_default_positives: bool = True
        self.is_default_negatives: bool = True

        self.default_positive_prompt: dict[str, list[str]] = {}
        self.default_negative_prompt: dict[str, list[str]] = {}

        self.get_default_prompts()

        self.negative_embeddings: dict[str, dict[str, str]] = {}
        self.get_negative_embeddings()

        self.is_show_prompts = True

        self.loras: list[str] = []
        self.lycos: list[str] = []

        self.config = self.read_config_file()

        logging_level = logging.DEBUG if self.config["logging"] else logging.INFO

        self.models: dict[str, Model] = load_models()

        logger.setLevel(logging_level)

        logger.info(f"Config: {self.config}")

    def read_config_file(self):
        config = {}
        with open(f"{scripts.basedir()}/config.json", "r") as f:
            config = json.load(f)

        return config

    def load_loras_and_lycos(self, path_to_models: str):
        lyco_path: str = f"{path_to_models}/LyCORIS"
        lora_path: str = f"{path_to_models}/Lora"

        self.loras = os.listdir(lora_path)
        self.lycos = os.listdir(lyco_path)

    def register_networks(self):
        loras_used: list[str] = []
        lycos_used: list[str] = []

        for model_name, model in self.models.items():
            for lora in self.loras:
                if check_valid_model_extension(lora) is False:
                    logger.debug(f"Skipping {lora} as it is not a model file")
                    continue

                lora = strip_model_file_extensions(lora)
                if model_name.lower() in lora.lower():
                    if model.mtype is None:
                        logger.warning(
                            f"{model_name} has no type specified. Setting {model_name} type to lora."
                        )
                        model.mtype = "lora"

                    mtype = model.mtype

                    if mtype.lower() != "lora":
                        logger.warning(
                            f"Model type does not match (lora) for {model_name}."
                        )
                        model.mtype = "lora"

                    model.model_path = lora
                    loras_used.append(lora)

            for lyco in self.lycos:
                if check_valid_model_extension(lyco) is False:
                    logger.debug(f"Skipping {lyco} as it is not a model file")
                    continue

                lyco = strip_model_file_extensions(lyco)
                if model_name.lower() in lyco.lower():
                    if model.mtype is None:
                        logger.warning(
                            f"{model_name} has no type specified. Setting {model_name} type to lyco."
                        )
                        model.mtype = "lyco"

                    mtype = model.mtype

                    if mtype.lower()[:4] != "lyco":
                        logger.warning(
                            f"Model type does not match (lyco) for {model_name}."
                        )
                        model.mtype = "lyco"

                    model.model_path = lyco
                    lycos_used.append(lyco)

            if model.model_path == None:
                logger.info(f"could not find any model for {model_name}.")

                if self.config["download"]:
                    logger.info(f"Trying to download {model_name} from {model.url}.")

                    if model.mtype is None:
                        logger.warning(
                            f"{model_name} has no type specified. Don't know where to download it to."
                        )
                    else:
                        success: bool = download_model(
                            model.url, model_name, model.mtype
                        )

                        if success:
                            model.model_path = f"{model.mtype}__{model_name}"

        for lora in self.loras:
            lora = strip_model_file_extensions(lora)
            if lora not in loras_used:
                logger.debug(f"Unused lora: {lora}")

        for lyco in self.lycos:
            lyco = strip_model_file_extensions(lyco)
            if lyco not in lycos_used:
                logger.debug(f"Unused lyco: {lyco}")

    def get_default_prompts(self):
        default_prompts_path = (
            f"{scripts.basedir()}/default_prompts/default_prompts.json"
        )

        if not os.path.exists(default_prompts_path):
            logging.warning(
                f"Could not find default prompts at {default_prompts_path}."
            )
            return

        with open(default_prompts_path, "r") as f:
            default_prompts = json.load(f)

        logging.debug(f"Default prompts: {default_prompts}")

        self.default_positive_prompt["default"] = default_prompts["positive"]

        self.default_negative_prompt["default"] = default_prompts["negative"]

    def get_negative_embeddings(self):
        embeddings_dir = f"{root_dir}/embeddings"

        if not os.path.exists(embeddings_dir):
            logger.warning(f"Could not find embeddings directory at {embeddings_dir}.")
            return

        embedding_files = os.listdir(embeddings_dir)

        embeddings_information = f"{base_dir}/embeddings/embeddings.json"

        if not os.path.exists(embeddings_information):
            logger.warning(
                f"Could not find embeddings information file at {embeddings_information}."
            )
            return

        with open(embeddings_information, "r") as f:
            embeddings = json.load(f)

        for embedding_name, embedding in embeddings.items():
            if embedding["file"] in embedding_files:
                self.negative_embeddings[embedding_name] = deepcopy(embedding)


def strip_model_file_extensions(model_name: str) -> str:
    return (
        model_name.replace(".ckpt", "").replace(".safetensors", "").replace(".pt", "")
    )


def check_valid_model_extension(model_name: str) -> bool:
    return (
        model_name.endswith(".ckpt")
        or model_name.endswith(".safetensors")
        or model_name.endswith(".pt")
    )
