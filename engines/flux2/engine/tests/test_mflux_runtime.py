from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from flux2_engine.mflux_runtime import (
    configure_text_length,
    generation_kwargs,
    model_config_for,
)


@dataclass
class Request:
    seed: int = 42
    prompt: str = "hello"
    resolved_steps: int = 4
    width: int = 512
    height: int = 512
    guidance: float = 1.0
    dynamic_text_length: bool = True
    image_path: Path | None = None
    image_paths: tuple[Path, ...] = ()
    image_strength: float = 0.75


class CurrentModel:
    def generate_image(
        self, seed, prompt, num_inference_steps, width, height, guidance
    ):
        pass


class LegacyModel:
    def generate_image(
        self, seed, prompt, num_inference_steps, width, height, guidance,
        dynamic_text_length,
    ):
        pass


class Tokenizer:
    padding = "max_length"


class ModelWithTokenizer(CurrentModel):
    tokenizers = {"qwen3": Tokenizer()}


def test_generation_kwargs_support_current_mflux_api():
    kwargs = generation_kwargs(CurrentModel(), Request())
    assert "dynamic_text_length" not in kwargs


def test_generation_kwargs_preserves_legacy_dynamic_text_length():
    kwargs = generation_kwargs(LegacyModel(), Request())
    assert kwargs["dynamic_text_length"] is True


def test_new_mflux_tokenizer_keeps_dynamic_text_contract():
    model = ModelWithTokenizer()
    configure_text_length(model, True)
    assert model.tokenizers["qwen3"].padding == "longest"
    configure_text_length(model, False)
    assert model.tokenizers["qwen3"].padding == "max_length"


class ModelConfigFactories:
    @staticmethod
    def flux2_klein_4b():
        return "4b"

    @staticmethod
    def flux2_klein_9b():
        return "9b"

    @staticmethod
    def flux2_klein_9b_kv():
        return "9b-kv"


class Symbols:
    model_config = ModelConfigFactories


def test_model_config_for_selects_resolved_variant():
    config = type(
        "Config",
        (),
        {"resolved_model_variant": type("Variant", (), {"value": "flux2-klein-9b"})()},
    )()
    assert model_config_for(Symbols(), config) == "9b"


class Img2ImgModel:
    def generate_image(
        self, seed, prompt, num_inference_steps, width, height, guidance,
        image_path=None, image_strength=None,
    ):
        pass


class EditModel:
    def generate_image(
        self, seed, prompt, num_inference_steps, width, height, guidance,
        image_paths=None,
    ):
        pass


def test_generation_kwargs_maps_img2img_path_and_strength():
    request = Request(image_path=Path("init.png"), image_strength=0.35)
    kwargs = generation_kwargs(Img2ImgModel(), request)
    assert kwargs["image_path"] == Path("init.png")
    assert kwargs["image_strength"] == 0.35


def test_generation_kwargs_maps_ordered_edit_references():
    request = Request(image_paths=(Path("one.png"), Path("two.png")))
    kwargs = generation_kwargs(EditModel(), request)
    assert kwargs["image_paths"] == [Path("one.png"), Path("two.png")]
