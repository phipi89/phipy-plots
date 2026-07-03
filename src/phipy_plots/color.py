"""Color classes, named color instances, and a discrete matplotlib colormap."""

from __future__ import annotations

import colorsys

import colorhash
import numpy as np
from colorutils import Color as ColorUtilsColor
from matplotlib.colors import to_hex, to_rgb

__all__ = [
    "Color",
    "Colors",
    "reduce_opacity",
    "discrete_cmap",
    "black",
    "gray",
    "yellow",
    "orange",
    "white",
    "colors",
    "map_colors",
    "blue",
    "red",
    "green",
]

_PLOTLY_DEFAULT_COLORS = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]

_MAP_PALETTE = [
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#d9d9d9",
    "#bc80bd",
    "#ccebc5",
    "#ffed6f",
]

_PALETTES = {
    "plot": _PLOTLY_DEFAULT_COLORS,
    "map": _MAP_PALETTE,
}


def _is_rgb_sequence(color):
    if isinstance(color, str) or not hasattr(color, "__iter__"):
        return False

    values = tuple(color)
    if len(values) not in (3, 4):
        return False

    return all(isinstance(v, (int, float, np.integer, np.floating)) for v in values)


def _normalize_color(color):
    if isinstance(color, Color):
        return color.hex

    if isinstance(color, str):
        if "rgba" in color:
            rgba = [float(v) for v in color.strip("rgba()").split(",")]
            color = tuple(v / 255 if i < 3 and v > 1 else v for i, v in enumerate(rgba))
            return to_hex(color)

        if "rgb" in color:
            rgb = [float(v) for v in color.strip("rgb()").split(",")]
            color = tuple(v / 255 if v > 1 else v for v in rgb)
            return to_hex(color)

        return to_hex(color)

    if hasattr(color, "__iter__"):
        color = tuple(float(v) for v in color)
        color = tuple(v / 255 if v > 1 else v for v in color)
        return to_hex(color)

    return to_hex(color)


class Color(str):
    def __new__(cls, color):
        return str.__new__(cls, _normalize_color(color))

    def __init__(self, color):
        self.hex = str(self)

    def alpha(self, alpha):
        return reduce_opacity(self, alpha)

    def lighter(self, factor=0.5):
        r, g, b = self.rgb()
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)

        return Color(ColorUtilsColor(rgb=(r, g, b)).hex)

    def blend(self, color, ratio=0.5):
        if not 0 <= ratio <= 1:
            raise ValueError("ratio must be between 0 and 1")

        h, s, v = colorsys.rgb_to_hsv(*self.rgb(fraction=True))
        other_h, other_s, other_v = colorsys.rgb_to_hsv(
            *Color(color).rgb(fraction=True)
        )

        if s == 0:
            h = other_h
        elif other_s == 0:
            other_h = h

        if v == 0:
            s = other_s
        elif other_v == 0:
            other_s = s

        dh = (other_h - h + 0.5) % 1 - 0.5
        blended_h = (h + ratio * dh) % 1
        blended_s = s * (1 - ratio) + other_s * ratio
        blended_v = v * (1 - ratio) + other_v * ratio
        blended = np.array(colorsys.hsv_to_rgb(blended_h, blended_s, blended_v))
        blended = (255 * blended).round().astype(int)

        return Color(ColorUtilsColor(rgb=tuple(blended)).hex)

    def darker(self, factor=2):
        h, s, v = colorsys.rgb_to_hsv(*to_rgb(self.hex))
        return Color(colorsys.hsv_to_rgb(h, s, v ** (2 * factor))).hex

    def rgb(self, fraction=False):
        rgb = tuple(round(v * 255) for v in to_rgb(self.hex))
        if fraction:
            return [v / 255 for v in rgb]
        else:
            return rgb


class Colors(list):
    fallback = Color("#CFCCC0")

    def __init__(self, color_list=None, recursive=True, kind=None):
        if color_list is not None:
            base = [color_list] if _is_rgb_sequence(color_list) else list(color_list)
        elif kind is not None:
            base = list(_PALETTES[kind])
        else:
            base = list(_PLOTLY_DEFAULT_COLORS)

        super().__init__(Color(c) for c in base)

        self.grayscale = [reduce_opacity(black, 1 / i) for i in range(1, 11)]

        if recursive:
            self.alt = Colors(
                [
                    Color("#13222A"),
                    Color("#5C8FBD"),
                    Color("#70B3A1"),
                    Color("#EDC032"),
                    Color("#E89027"),
                ],
                recursive=False,
            )

        self.first = reduce_opacity(black, 0.75)
        self.second = reduce_opacity(black, 0.5)
        self.third = reduce_opacity(black, 0.25)

        self.A = Color("#E89287")
        self.B = Color("#7BA3D1")
        self.C = Color("#A5DAC5")

    @property
    def discrete(self):
        return self

    def __getitem__(self, index):
        if isinstance(index, bool):
            if index:
                return list.__getitem__(self, 0)
            else:
                return list.__getitem__(self, 1)

        if isinstance(index, (int, np.integer)):
            try:
                return list.__getitem__(self, index)
            except IndexError:
                return self.fallback

        try:
            return [self[i] for i in np.asarray(index).astype(int)]
        except (TypeError, ValueError):
            return list.__getitem__(self, index)

    def __call__(self, arg):
        if isinstance(arg, np.ndarray):
            if arg.dtype == np.bool_:
                return [self.A if b else self.B for b in arg]

            elif arg.dtype == np.int_:
                return [self[i] for i in arg]

        return self.from_hash(arg)

    def from_hash(self, hashable):
        return colorhash.ColorHash(hashable).hex

    def cmap(self, name="eartools", n=None, kind=None):
        from matplotlib.colors import ListedColormap

        if kind is not None:
            palette = Colors(kind=kind, recursive=False)
            color_list = palette.discrete[:n] if n is not None else palette.discrete
        else:
            color_list = self.discrete[:n] if n is not None else self.discrete
        return ListedColormap([c.hex for c in color_list], name=name, N=len(color_list))


def reduce_opacity(color, opacity=0.5):
    if isinstance(color, int):
        color = Color(colors[color])
    else:
        color = Color(color)

    return "rgba({}, {}, {}, {})".format(*color.rgb() + (opacity,))


def discrete_cmap(name="eartools", n=None, color_list=None, kind=None):
    if color_list is not None:
        return Colors(color_list=color_list, recursive=False).cmap(name=name, n=n)
    if kind is not None:
        return Colors(kind=kind, recursive=False).cmap(name=name, n=n)
    return colors.cmap(name=name, n=n)


black = Color("#000000")
gray = Color("#999999")
yellow = Color("#FFD966")
orange = Color("#FFA07A")
white = Color("#FFFFFF")
colors = Colors()
map_colors = Colors(kind="map")
blue, red, green, *_ = colors
