"""Comfy-specific type hinting"""

from __future__ import annotations
from typing import Literal, TypedDict, Optional, Any
from typing_extensions import NotRequired
from abc import ABC, abstractmethod
from enum import Enum


class StrEnum(str, Enum):
    """Base class for string enums. Python's StrEnum is not available until 3.11."""

    def __str__(self) -> str:
        return self.value


class IO(StrEnum):
    """Node input/output data types.

    Includes functionality for ``"*"`` (`ANY`) and ``"MULTI,TYPES"``.
    """

    STRING = "STRING"
    IMAGE = "IMAGE"
    MASK = "MASK"
    LATENT = "LATENT"
    BOOLEAN = "BOOLEAN"
    INT = "INT"
    FLOAT = "FLOAT"
    COMBO = "COMBO"
    CONDITIONING = "CONDITIONING"
    SAMPLER = "SAMPLER"
    SIGMAS = "SIGMAS"
    GUIDER = "GUIDER"
    NOISE = "NOISE"
    CLIP = "CLIP"
    CONTROL_NET = "CONTROL_NET"
    VAE = "VAE"
    MODEL = "MODEL"
    LORA_MODEL = "LORA_MODEL"
    LOSS_MAP = "LOSS_MAP"
    CLIP_VISION = "CLIP_VISION"
    CLIP_VISION_OUTPUT = "CLIP_VISION_OUTPUT"
    STYLE_MODEL = "STYLE_MODEL"
    GLIGEN = "GLIGEN"
    UPSCALE_MODEL = "UPSCALE_MODEL"
    AUDIO = "AUDIO"
    WEBCAM = "WEBCAM"
    POINT = "POINT"
    FACE_ANALYSIS = "FACE_ANALYSIS"
    BBOX = "BBOX"
    SEGS = "SEGS"
    VIDEO = "VIDEO"

    ANY = "*"
    """Always matches any type, but at a price.

    Causes some functionality issues (e.g. reroutes, link types), and should be avoided whenever possible.
    """
    NUMBER = "FLOAT,INT"
    """A float or an int - could be either"""
    PRIMITIVE = "STRING,FLOAT,INT,BOOLEAN"
    """Could be any of: string, float, int, or bool"""

    def __ne__(self, value: object) -> bool:
        if self == "*" or value == "*":
            return False
        if not isinstance(value, str):
            return True
        a = frozenset(self.split(","))
        b = frozenset(value.split(","))
        return not (b.issubset(a) or a.issubset(b))


class RemoteInputOptions(TypedDict):
    route: str
    """The route to the remote source."""
    refresh_button: bool
    """Specifies whether to show a refresh button in the UI below the widget."""
    control_after_refresh: Literal["first", "last"]
    """Specifies the control after the refresh button is clicked. If "first", the first item will be automatically selected, and so on."""
    timeout: int
    """The maximum amount of time to wait for a response from the remote source in milliseconds."""
    max_retries: int
    """The maximum number of retries before alert aborting the request."""
    refresh_ms: NotRequired[int] # renamed to avoid potential name collisions or clarity


class MultiSelectOptions(TypedDict):
    placeholder: NotRequired[str]
    """The placeholder text to display in the multi-select widget when no items are selected."""
    chip: NotRequired[bool]
    """Specifies whether to use chips instead of comma separated values for the multi-select widget."""


class InputTypeOptions(TypedDict):
    """Provides type hinting for the return type of the INPUT_TYPES node function.
    """

    default: NotRequired[bool | str | float | int | list | tuple]
    forceInput: NotRequired[bool]
    lazy: NotRequired[bool]
    rawLink: NotRequired[bool]
    tooltip: NotRequired[str]
    socketless: NotRequired[bool]
    widgetType: NotRequired[str]
    min: NotRequired[float]
    max: NotRequired[float]
    step: NotRequired[float]
    round: NotRequired[float]
    label_on: NotRequired[str]
    label_off: NotRequired[str]
    multiline: NotRequired[bool]
    placeholder: NotRequired[str]
    dynamicPrompts: NotRequired[bool]
    image_upload: NotRequired[bool]
    image_folder: NotRequired[Literal["input", "output", "temp"]]
    remote: NotRequired[RemoteInputOptions]
    control_after_generate: NotRequired[bool]
    options: NotRequired[list[str | int | float]]
    multi_select: NotRequired[MultiSelectOptions]


class HiddenInputTypeDict(TypedDict):
    """Provides type hinting for the hidden entry of node INPUT_TYPES."""

    node_id: NotRequired[Literal["UNIQUE_ID"]]
    unique_id: NotRequired[Literal["UNIQUE_ID"]]
    prompt: NotRequired[Literal["PROMPT"]]
    extra_pnginfo: NotRequired[Literal["EXTRA_PNGINFO"]]
    dynprompt: NotRequired[Any]


class InputTypeDict(TypedDict):
    """Provides type hinting for node INPUT_TYPES."""

    required: NotRequired[dict[str, tuple[IO, InputTypeOptions]]]
    optional: NotRequired[dict[str, tuple[Any, InputTypeOptions]]] # looser typing for custom types
    hidden: NotRequired[HiddenInputTypeDict]


class ComfyNodeABC(ABC):
    """Abstract base class for Comfy nodes."""

    DESCRIPTION: str
    CATEGORY: str
    EXPERIMENTAL: bool
    DEPRECATED: bool
    API_NODE: Optional[bool]

    @classmethod
    @abstractmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {"required": {}}

    OUTPUT_NODE: bool
    INPUT_IS_LIST: bool
    OUTPUT_IS_LIST: tuple[bool, ...]
    RETURN_TYPES: tuple[IO | str, ...]
    RETURN_NAMES: tuple[str, ...]
    OUTPUT_TOOLTIPS: tuple[str, ...]
    FUNCTION: str


class CheckLazyMixin:
    """Provides a basic check_lazy_status implementation."""

    def check_lazy_status(self, **kwargs) -> list[str]:
        need = [name for name in kwargs if kwargs[name] is None]
        return need


class FileLocator(TypedDict):
    """Provides type hinting for the file location"""

    filename: str
    subfolder: str
    type: Literal["input", "output", "temp"]
