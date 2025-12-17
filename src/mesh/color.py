import numpy as np
from omegaconf import DictConfig


def get_color(config: DictConfig, num_colors: int = 1) -> dict[str, np.ndarray]:
    start_color = np.array(config["start_color"])
    end_color = np.array(config["end_color"]) if config["end_color"] else start_color
    if start_color[:3].max() > 1:
        start_color[:3] = start_color[:3] / 255
    if end_color[:3].max() > 1:
        end_color = end_color / 255

    start_hsv = np.array(config["start_hsv"])
    end_hsv = np.array(config["end_hsv"]) if config["end_hsv"] else start_hsv

    start_brightness = np.array(config["start_brightness"])
    end_brightness = (
        np.array(config["end_brightness"])
        if config["end_brightness"]
        else start_brightness
    )

    start_contrast = np.array(config["start_contrast"])
    end_contrast = (
        np.array(config["end_contrast"]) if config["end_contrast"] else start_contrast
    )

    return {
        "RGBA": np.linspace(start_color, end_color, num_colors),
        "HSV": np.linspace(start_hsv, end_hsv, num_colors),
        "brightness": np.linspace(start_brightness, end_brightness, num_colors),
        "contrast": np.linspace(start_contrast, end_contrast, num_colors),
    }
