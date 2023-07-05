import json
import os

"""
This script adds the type information to the model information files. Based on where the model was found, it will be marked as either LoRA or LyCORIS.
This is for models that were downloaded before the model information files were created.
"""

root_dir = ""  # automatic root
model_relative_dir = "models"
lora_relative_dir = "Lora"
lyco_relative_dir = "LyCORIS"
model_information_relative_path = "extensions/lora_collector/models"

loras = os.listdir(f"{root_dir}/{model_relative_dir}/{lora_relative_dir}")
lycos = os.listdir(f"{root_dir}/{model_relative_dir}/{lyco_relative_dir}")


for file in os.listdir(f"{model_information_relative_path}"):
    if file.endswith(".json"):
        full_file_path = f"{root_dir}/{model_information_relative_path}/{file}"
        with open(full_file_path, "r") as f:
            data = json.load(f)

            for model_name, model in data.items():
                for lora in loras:
                    if model_name.lower() in lora.lower():
                        ntype = "LoRA"
                        model["type"] = ntype

                for lyco in lycos:
                    if model_name.lower() in lyco.lower():
                        ntype = "LyCORIS"
                        model["type"] = ntype

            with open(full_file_path, "w") as f:
                json.dump(data, f, indent=2)
