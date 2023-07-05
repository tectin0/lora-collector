## LoRA Collector

This is an extension for the [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) by AUTOMATIC1111.

# Description

LoRA Collector is an extension for the Stable Diffusion WebUI by AUTOMATIC1111. This extension serves as a personal collection of different LoRAs and LyCORIS models sourced from the civitai.com website.

The collection of LoRAs and LyCORIS models has been gathered by hand and stored in a JSON file, which allows for efficient categorization and organization. Each model comes with its corresponding URL and prompts. Models are downloaded if ` "download"` is `true` in the `config.json` file. The models are stored in the corresponding "models" folder (Lora, LyCORIS). New models can be added by adding a new entry to the JSON file. The JSON file is automatically read when the extension is loaded.

Moreover, LoRA Collector supports the usage of embeddings. If there are embedding files available in the designated "embeddings" folder and they are properly listed in the "embeddings.json" file.

There are also some default prompts that are included in the extension. These prompts are stored in the "default_prompts" folder. They are enabled by default and can be disabled or modified in the "Basic" category.

This extension provides functionality to enable/disable specific models and choose which prompts to use in the WebUI.

# Installation

To install LoRA Collector for the Stable Diffusion WebUI by AUTOMATIC1111, follow these steps:

1. Clone this repository to the extensions folder in the AUTOMATIC repository. You can do so by running the following command:
   `git clone https://github.com/tectin0/lora-collector.git path/to/AUTOMATIC/extensions/lora_collector`

2. Install the corresponding [LyCORIS extension](https://github.com/KohakuBlueleaf/a1111-sd-webui-lycoris). Please refer to the documentation of the LyCORIS extension for installation instructions.

Without any models downloaded, the initial startup might take a while.

# Usage

## DISCLAIMER: The quality of the generated outputs is dependent on the quality of the models and prompts. The models and prompts are sourced from the civitai.com website. The models and prompts are not guaranteed to be of high quality. Please use at your own discretion.

To use LoRA Collector within the Stable Diffusion WebUI, follow these steps:

1. Open the Stable Diffusion WebUI by AUTOMATIC1111.

2. Navigate to either the "txt2img" or "img2img" tab.

3. Within the selected tab, you will find a new accordion titled "LoRA Collector". Click on the accordion to expand it.

4. Inside the "LoRA Collector" accordion, you will see different categories corresponding to the JSON files containing the models and prompts.

5. Click on the checkbox next to each model to display the strength slider and the prompts associated with that model. By default, some prompts may be activated for the models, while others may be optional.

6. To generate outputs utilizing a specific model and prompts, ensure that the checkbox for that model is checked. The activated prompts will be added automatically when generating results.

7. Adjust the strength slider according to your desired level of influence on the generated outputs.

8. Proceed with the text-to-image or image-to-image generation process using the selected models and prompts.

# Contributing

This is a personal project. I do not intend to maintain this project. However, if you would like to contribute, feel free to open a pull request.

# Credits

- [Stable Diffusion WebUI by AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

- [LyCORIS extension by KohakuBlueleaf](https://github.com/KohakuBlueleaf/a1111-sd-webui-lycoris)
