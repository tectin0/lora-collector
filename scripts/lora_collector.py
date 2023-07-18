from copy import deepcopy
import logging

logger = logging.getLogger("lora_collector")

from typing import Any, Optional, Tuple
import gradio as gr

import sys
import os
import modules.scripts as scripts  # type: ignore

base_dir = scripts.basedir()
model_path = (
    f"{os.sep.join(os.path.normpath(base_dir).split(os.sep)[:-2])}{os.sep}models"
)
sys.path.append(base_dir)


from lib_lora_collector.lora_manager import LoraManager
from lib_lora_collector.models import Model
from lib_lora_collector.typing import StableDiffusionProcessing

logger.info("lora collector loaded")

manager = LoraManager()

manager.load_loras_and_lycos(model_path)

manager.register_networks()

paste_field_names: list[str] = []  # TODO: add for other fields?
infotext_fields: list[Tuple[Any, str]] = []


class Scripts(scripts.Script):
    def title(self):
        return "lora collector"

    def show(
        self,
        is_img2img: bool,
    ):
        return scripts.AlwaysVisible

    def ui(
        self,
        is_img2img: bool,
    ):
        self.paste_field_names: list[str] = []
        self.infotext_fields: list[Tuple[Any, str]] = []

        with gr.Group():
            main_accordion = gr.Accordion("LoRA Collector", open=False)

            with main_accordion:
                is_enabled = gr.Checkbox(label="Enabled", value=False)

                def on_enabled(checkbox):
                    manager.is_enabled = checkbox

                    return gr.Accordion.update(open=checkbox)

                is_enabled.change(
                    on_enabled,
                    inputs=[is_enabled],
                    outputs=[main_accordion],
                )

                model_categories: list[str] = list(
                    set([model.category for model in manager.models.values()])
                )

                model_categories.sort(key=lambda x: x.lower())

                model_subcategories: list[str] = list(
                    set(
                        [
                            f"{model.category}|{model.subcategory}"
                            for model in manager.models.values()
                            if model.subcategory
                        ]
                    )
                )

                def make_model_entry(model_name: str, model: Model):
                    model.prompt = ", ".join(model.prompts)
                    model.negative_prompt = ", ".join(model.negative_prompts)

                    visible_name = model_name.replace("_", " ")

                    checkbox = gr.Checkbox(label=visible_name, value=False)
                    label = gr.Label(model_name, visible=False)
                    slider = gr.Slider(
                        minimum=-2.0,
                        maximum=2.0,
                        value=0.0,
                        visible=False,
                    )

                    prompts = gr.Dropdown(
                        choices=model.prompts + model.optional_prompts,
                        value=deepcopy(manager.models[deepcopy(model_name)].prompts),
                        label=f"{visible_name} prompts",
                        multiselect=True,
                        interactive=True,
                        visible=False,
                    )

                    negative_prompts = gr.Dropdown(
                        choices=model.negative_prompts,
                        value=deepcopy(
                            manager.models[deepcopy(model_name)].negative_prompts
                        ),
                        label=f"{visible_name} negative prompts",
                        multiselect=True,
                        interactive=True,
                        visible=False,
                    )

                    prompts.change(
                        on_prompts_change,
                        inputs=[label, prompts, negative_prompts],
                        outputs=[],
                    )

                    negative_prompts.change(
                        on_prompts_change,
                        [label, prompts, negative_prompts],
                        outputs=[],
                    )

                    checkbox.select(
                        on_checkbox_selected,
                        inputs=[
                            checkbox,
                            label,
                            slider,
                        ],
                        outputs=[
                            slider,
                            prompts,
                            negative_prompts,
                        ],
                    )

                    slider.change(
                        on_slider_change,
                        inputs=[label, slider],
                        outputs=[],
                    )

                with gr.Tab("Basics"):
                    checkbox_default_positives = gr.Checkbox(
                        label="Default Positives", value=True
                    )

                    dropdown_default_positives = gr.Dropdown(
                        label="Positive Prompts",
                        choices=manager.default_positive_prompt["default"],
                        value=lambda: manager.default_positive_prompt["default"],
                        multiselect=True,
                        interactive=True,
                        visible=True,
                    )

                    def on_default_positives_checkbox(
                        checkbox,
                        dropdown,
                    ) -> dict[str, Any]:
                        manager.is_default_positives = checkbox

                        if checkbox and dropdown is not None:
                            manager.default_positive_prompt["default"] = dropdown
                        else:
                            manager.default_positive_prompt["default"] = []

                        return gr.Dropdown.update(visible=checkbox)

                    checkbox_default_positives.change(
                        on_default_positives_checkbox,
                        inputs=[
                            checkbox_default_positives,
                            dropdown_default_positives,
                        ],
                        outputs=[dropdown_default_positives],
                    )

                    dropdown_default_positives.change(
                        on_default_positives_checkbox,
                        inputs=[
                            checkbox_default_positives,
                            dropdown_default_positives,
                        ],
                        outputs=[],
                    )

                    def on_default_negatives(
                        checkbox,
                        dropdown,
                    ) -> dict[str, Any]:
                        manager.is_default_negatives = checkbox

                        if checkbox and dropdown is not None:
                            manager.default_negative_prompt["default"] = dropdown
                        else:
                            manager.default_negative_prompt["default"] = []

                        return gr.Dropdown.update(visible=checkbox)

                    checkbox_default_negatives = gr.Checkbox(
                        label="Default Negatives", value=True
                    )

                    dropdown_default_negatives = gr.Dropdown(
                        label="Negative Prompts",
                        choices=manager.default_negative_prompt["default"],
                        value=lambda: manager.default_negative_prompt["default"],
                        multiselect=True,
                        interactive=True,
                        visible=True,
                    )

                    checkbox_default_negatives.change(
                        on_default_negatives,
                        inputs=[
                            checkbox_default_negatives,
                            dropdown_default_negatives,
                        ],
                        outputs=[dropdown_default_negatives],
                    )

                    dropdown_default_negatives.change(
                        on_default_negatives,
                        inputs=[
                            checkbox_default_negatives,
                            dropdown_default_negatives,
                        ],
                        outputs=[],
                    )

                    for negative_embedding_name in manager.negative_embeddings.keys():
                        checkbox_negative_embedding = gr.Checkbox(
                            label=negative_embedding_name,
                            value=False,
                        )

                        label_negative_embedding = gr.Label(
                            negative_embedding_name,
                            visible=False,
                        )

                        def on_negative_embedding(checkbox, label_container):
                            label: str = label_container["label"]

                            if checkbox:
                                manager.default_negative_prompt[label] = [
                                    manager.negative_embeddings[label]["prompt"]
                                ]
                            else:
                                manager.default_negative_prompt[
                                    negative_embedding_name
                                ] = []

                        checkbox_negative_embedding.change(
                            on_negative_embedding,
                            inputs=[
                                checkbox_negative_embedding,
                                label_negative_embedding,
                            ],
                            outputs=[],
                        )

                    if "detail_tweaker" in manager.models:
                        make_model_entry(
                            "detail_tweaker", manager.models["detail_tweaker"]
                        )

                for category in model_categories:
                    if category == "Basic":
                        continue

                    with gr.Tab(category):
                        for network_name, network in manager.models.items():
                            if network.category != category:
                                continue

                            if network.subcategory is None:
                                make_model_entry(network_name, network)

                        for subcategory in model_subcategories:
                            category_comp, subcategory = subcategory.split("|")

                            if not category == category_comp:
                                continue

                            with gr.Accordion(subcategory, open=False):
                                for network_name, network in manager.models.items():
                                    if network.category != category:
                                        continue

                                    if (
                                        network.subcategory is not None
                                        and network.subcategory == subcategory
                                    ):
                                        make_model_entry(network_name, network)

                    show_prompts_checkbox: Optional[gr.Checkbox] = None

                with gr.Tab("Config"):
                    show_prompts_checkbox = gr.Checkbox(
                        label="Show Prompts",
                        value=True,
                    )

                    infotext_fields.append((show_prompts_checkbox, "show_prompts"))
                    paste_field_names.append("show_prompts")

                    def on_show_prompts_checkbox(checkbox):
                        manager.is_show_prompts = checkbox

                    show_prompts_checkbox.change(
                        on_show_prompts_checkbox,
                        inputs=[show_prompts_checkbox],
                        outputs=[],
                    )

    def process(
        self,
        p: StableDiffusionProcessing,
        *args,
    ):
        if not manager.is_enabled:
            return

        for batch_index, _ in enumerate(p.all_prompts):
            self.prompting(p, batch_index)

    def prompting(
        self,
        p: StableDiffusionProcessing,
        batch_index: int,
    ):
        original_prompt: str = p.all_prompts[batch_index]
        original_negative_prompt: str = p.all_negative_prompts[batch_index]

        default_positive_prompt: str = ""
        default_negative_prompt: str = ""

        for _, prompt_list in manager.default_positive_prompt.items():
            if prompt_list == []:
                continue

            default_positive_prompt += ", ".join(prompt_list)
            default_positive_prompt += ", "

        for _, prompt_list in manager.default_negative_prompt.items():
            if prompt_list == []:
                continue

            default_negative_prompt += ", ".join(prompt_list)
            default_negative_prompt += ", "

        if default_positive_prompt != "":
            original_prompt = default_positive_prompt + original_prompt

        if default_negative_prompt != "":
            original_negative_prompt = (
                default_negative_prompt + original_negative_prompt
            )

        for model_name, model in manager.models.items():
            if not model.model_path:
                logger.debug(f"Skipping {model_name} as it has no models")
                continue

            if model.strength != 0.0:
                model_prompt = model.prompt
                model_negative_prompt = model.negative_prompt

                model_type = model.mtype

                if model_type is None:
                    logger.debug(f"Skipping {model_name} as it has no type")
                    continue

                model_type = model_type[:4].lower()

                model_path = model.model_path
                strength = model.strength

                original_prompt += (
                    f", {model_prompt} <{model_type}:{model_path}:{strength}>"
                )

                original_negative_prompt += f", {model_negative_prompt}"

        p.all_prompts[batch_index] = original_prompt
        p.all_negative_prompts[batch_index] = original_negative_prompt


def on_checkbox_selected(
    checkbox: bool,
    label_container: dict[str, str],
    slider: float,
) -> Tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    label = label_container["label"]
    is_negative_prompts = manager.models[label].negative_prompts != []

    return (
        gr.Slider.update(visible=checkbox, value=0.0 if not checkbox else slider),
        gr.Dropdown.update(visible=(manager.is_show_prompts and checkbox)),
        gr.Dropdown.update(
            visible=(manager.is_show_prompts and checkbox and is_negative_prompts)
        ),
    )


def on_slider_change(
    label_container: dict[str, str],
    style_strength_slider: float,
):
    label: str = label_container["label"]
    manager.models[label].strength = style_strength_slider


def on_prompts_change(
    label_container: dict[str, str],
    prompts: list[str],
    negative_prompts: list[str],
):
    label: str = label_container["label"]
    manager.models[label].prompt = ", ".join(prompts)
    manager.models[label].negative_prompt = ", ".join(negative_prompts)
