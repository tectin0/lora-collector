from copy import deepcopy
import json
import logging
import os
from typing import Optional

root_path = os.sep.join(os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-1])


class Model:
    def __init__(
        self,
        url: str,
        mtype: Optional[str] = None,
        category: str = "Misc",
        subcategory: Optional[str] = None,
        prompts: Optional[list[str]] = None,
        optional_prompts: Optional[list[str]] = None,
        negative_prompts: Optional[list[str]] = None,
        model_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        prompts = prompts if prompts is not None else []
        optional_prompts = optional_prompts if optional_prompts is not None else []
        negative_prompts = negative_prompts if negative_prompts is not None else []

        self.url: str = url
        self.category: str = category
        self.subcategory: Optional[str] = subcategory
        self.mtype: Optional[str] = mtype
        self.prompts: list[str] = prompts
        self.optional_prompts: list[str] = optional_prompts
        self.negative_prompts: list[str] = negative_prompts
        self.model_path: Optional[str] = deepcopy(model_path)

        self.strength: float = 0.0
        self.prompt: str = ""
        self.negative_prompt: str = ""

    def toJSON(self) -> dict[str, str | list[str] | None]:
        return {
            "url": self.url,
            "category": self.category,
            "subcategory": self.subcategory,
            "prompts": self.prompts,
            "optional_prompts": self.optional_prompts,
        }


def load_models() -> dict[str, Model]:
    models: dict[str, Model] = dict()

    models_path = os.path.join(root_path, "models")

    for model_json in os.listdir(models_path):
        if not model_json.endswith(".json"):
            continue

        with open(os.path.join(models_path, model_json), "r") as f:
            data = json.load(f)

            for model_name, model in data.items():
                logging.debug(f"Loading model: {model_name}")

                url: str = str(model["url"]) if "url" in model else "None"
                ntype: Optional[str] = (
                    str(model["type"])
                    if "type" in model and model["type"] is not None
                    else None
                )
                category: str = (
                    str(model["category"])
                    if "category" in model and model["category"] is not None
                    else "Other"
                )

                subcategory: Optional[str] = (
                    str(model["subcategory"])
                    if "subcategory" in model and model["subcategory"] is not None
                    else "Other"
                )

                prompts: list[str] = (
                    list(model["prompts"])
                    if "prompts" in model and model["prompts"] is not None
                    else []
                )

                optional_prompts: list[str] = (
                    model["optional_prompts"]
                    if "optional_prompts" in model
                    and model["optional_prompts"] is not None
                    else []
                )

                negative_prompts: list[str] = (
                    model["negative_prompts"]
                    if "negative_prompts" in model
                    and model["negative_prompts"] is not None
                    else []
                )

                models[model_name] = Model(
                    url=url,
                    mtype=ntype,
                    category=category,
                    subcategory=subcategory,
                    prompts=prompts,
                    optional_prompts=optional_prompts,
                    negative_prompts=negative_prompts,
                )

    models = dict(sorted(models.items(), key=lambda item: item[0]))

    return models
