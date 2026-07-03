import io
import itertools
from types import MethodType

import colorlover as cl
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import scipy
from matplotlib.colors import cnames
from PIL import Image
from plotly.subplots import make_subplots

from phipy_plots.colors import (
    Color,
    Colors,
    black,
    blue,
    colors,
    gray,
    green,
    orange,
    red,
    reduce_opacity,
    white,
    yellow,
)

base = dict(
    template="plotly_white",
    margin=dict(l=10, r=10, t=10, b=10),
    font_family="Helvetica",
    font_color="black",
)

legend_topleft = dict(
    legend=dict(
        orientation="h",
        x=1,
        y=1.1,
        xanchor="right",
        yanchor="bottom",
        bordercolor="#eee",
        borderwidth=1,
    )
)

legend = legend_topleft

shading = dict(
    lighting=dict(specular=2, fresnel=0, diffuse=0.5, roughness=0.5, ambient=0.6)
)





def Figure(**kwargs):
    """Return a go.Figure object with added helper functions:
    - fig._xtitles : list of axis titles that get applied subsequently
                     to each subplot. Also available are `_ytitles` and
                     `_xytitles`.
    - fig._fit : add a fit to the last added trace (or the first of a
                 subplot, if `row`, `col` are defined). `type` can be
                 'polynomial', 'exp' or 'log'.
    - fig._strip : Don't show gridlines.
    """

    fig = make_subplots(**kwargs)

    fig._loop = lambda *args: loop(fig, *args)

    # NOTE: Attaching functions as here is not very elegant. Rather
    # do it using types.MethodType as done further below.
    fig._xtitles = lambda *args, **kwargs: add_titles(fig, "x", *args, **kwargs)
    fig._ytitles = lambda *args, **kwargs: add_titles(fig, "y", *args, **kwargs)
    fig._titles = lambda *args, **kwargs: add_titles(fig, "xy", *args, **kwargs)

    from functools import partial

    fig._point = MethodType(partial(add_curve, mode="markers"), fig)
    fig._curve = MethodType(add_curve, fig)
    fig._point = MethodType(add_point, fig)
    fig._points = MethodType(add_point, fig)
    fig._mesh = MethodType(add_mesh, fig)

    fig._fit = lambda **args: add_fit(fig, **args)

    fig._semilogx = lambda **kwargs: log_axis(fig, "x", **kwargs)
    fig._semilogy = lambda **kwargs: log_axis(fig, "y", **kwargs)
    fig._log = lambda: log_axis(fig, "xy")

    # keep this to not break things, but maybe rather use fig._equal
    fig._square = MethodType(square_axes, fig)
    fig._equal = MethodType(square_axes, fig)

    fig._strip = MethodType(strip_figure, fig)
    fig._strip_x = MethodType(lambda fig: strip_figure(fig, y=False), fig)
    fig._strip_y = MethodType(lambda fig: strip_figure(fig, x=False), fig)
    fig._highlight = MethodType(highlight_segment, fig)
    fig._legend = MethodType(_legend, fig)
    fig._indicate = MethodType(indicate, fig)
    fig._blend_scatter = MethodType(blend_scatter, fig)
    fig._radial_ticks = MethodType(radial_ticks, fig)

    fig._view = MethodType(set_view, fig)

    fig._rasterize = MethodType(rasterize, fig)
    fig._thumbnail = MethodType(thumbnail, fig)
    fig._large = MethodType(make_large, fig)
    fig._big = fig._large
    fig._wide = MethodType(make_wide, fig)

    i = 0  # orthographic projection for all scenes
    while True:
        try:
            scene = fig.layout[f"scene{i if i else ''}"]
            scene["camera"]["projection"] = dict(type="orthographic")
            i += 1
        except:
            break
    fig.update_layout(showlegend=False, template="plotly_white", **regular)
    fig.update_xaxes(
        gridcolor=reduce_opacity(black, 0.0675),
        zerolinecolor=reduce_opacity(black, 0.125),
    )
    fig.update_yaxes(
        gridcolor=reduce_opacity(black, 0.0675),
        zerolinecolor=reduce_opacity(black, 0.125),
    )

    fig._showlegend = lambda: fig.update_layout(showlegend=True)
    fig._title = MethodType(add_title, fig)

    fig.update_xaxes(mirror=True, linecolor=black.alpha(0.125), showline=True)
    fig.update_yaxes(mirror=True, linecolor=black.alpha(0.125), showline=True)
    fig.update_layout(**legend, legend_bordercolor=black.alpha(0.125), width=500)
    return fig


def thumbnail(fig):
    img = fig._rasterize()
    plt.figure(figsize=(150, 150), dpi=1)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def make_large(fig):
    fig.update_layout(width=1200, height=900)


def make_wide(fig):
    fig.update_layout(width=1200, height=600)


def set_view(fig, elev=None, azim=None, type="orthographic", r=2.0, scene=None):
    """Set 3d view using elevation and azimuth.
    This makes it mostly compatible with matplotlib's view settings.
    """

    if scene is None:
        try:
            scene = sorted([d.scene for d in fig.data])[-1][-1]
        except:
            scene = 1
        if scene == "e":
            scene = 1

    elev, azim = np.radians(elev), np.radians(azim)

    eye = dict(
        x=r * np.cos(elev) * np.cos(azim),
        y=r * np.cos(elev) * np.sin(azim),
        z=r * np.sin(elev),
    )

    fig.update_layout(
        {f"scene{scene}_camera": dict(eye=eye, projection=dict(type=type))}
    )


def indicate(fig, x=None, y=None, shift=0, color="black", row="all", col="all"):
    """Put an indicator on the x-axis at `x`."""

    if x:
        if not isinstance(x, (list, tuple, np.ndarray)):
            x = [x]
        y = [shift] * len(x)
        marker_symbol = 5
    elif y:
        if not isinstance(y, (list, tuple, np.ndarray)):
            y = [y]
        x = [shift] * len(y)
        marker_symbol = 8
    else:
        raise ValueError(
            "Need to set either `x` or `y` for vertical or horizontal indicator"
        )
    fig.add_scatter(
        x=x,
        y=y,
        marker_symbol=marker_symbol,
        marker_color=color,
        marker_line_color=color,
        marker_size=10,
        showlegend=False,
        row=row,
        col=col,
    )


def add_mesh(fig, mesh, shift=[0, 0, 0], shading="static", flat=False, **kwargs):
    """Assumes a trimesh.Trimesh and plots it as a mesh3d surface."""

    row = kwargs.pop("row", None)
    col = kwargs.pop("col", None)

    mesh = mesh.copy()

    if np.ndim(shift) == 0:
        mesh.vertices += mesh.vertex_normals * shift
    else:
        mesh.vertices += shift

    x, y, z = mesh.vertices.T
    i, j, k = mesh.faces.T

    if isinstance(shading, list) and len(shading) == 3:
        up_vector = shading
        shading = "static"
    else:
        up_vector = [0, 0, 1]

    if shading not in ("static", "white", "mask", "hide"):
        add_3d_trace(
            fig, go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, **kwargs), row=row, col=col
        )
        return

    if shading == "static":
        color = kwargs.pop("color", white)

        if not hasattr(color, "hex"):
            try:
                color = Color(cnames[color])
            except:
                raise ValueError(f"Unknown color `{color}`")

        colorscale = [[0, f"rgb{black.rgb()}"], [1, f"rgb{color.rgb()}"]]

    elif shading == "white":
        colorscale = [[0, f"rgb{white.rgb()}"], [1, f"rgb{white.rgb()}"]]

    elif shading == "mask":
        colorscale = [[0, f"rgb{black.rgb()}"], [1, f"rgb{black.rgb()}"]]

    elif shading == "hide":
        colorscale = [[0, f"rgb{black.rgb()}"], [1, f"rgb{black.rgb()}"]]
        kwargs["opacity"] = 0

    add_3d_trace(
        fig,
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            intensity=(
                mesh.face_normals.dot(up_vector)
                if flat
                else mesh.vertex_normals.dot(up_vector)
            ),
            intensitymode="cell" if flat else "vertex",
            colorscale=colorscale,
            cmin=-5,
            cmax=1.5,
            lighting=dict(ambient=1, diffuse=0, specular=0, roughness=1, fresnel=0),
            showscale=False,
            **kwargs,
        ),
        row=row,
        col=col,
    )


def sanitize_kwargs(kwargs, elements, prepend):
    for key in elements:
        if key in kwargs:
            kwargs[f"{prepend}{key}"] = kwargs.pop(key)
    return kwargs


def add_3d_trace(fig, trace, row=None, col=None):
    if row is None and col is None:
        fig.add_trace(trace)
        return

    if row is None:
        row = 1
    if col is None:
        col = 1

    subplot = fig._grid_ref[row - 1][col - 1][0]
    if subplot.subplot_type == "scene":
        fig.add_trace(trace, row=row, col=col)
        return

    if subplot.subplot_type != "xy":
        raise ValueError(
            f"3D traces are not compatible with subplot type {subplot.subplot_type!r}"
        )

    layout = np.array(fig._grid_ref, dtype="object").shape
    _, ncols, *_ = layout
    index = (row - 1) * ncols + col
    scene = f"scene{index if index > 1 else ''}"

    xaxis, yaxis = [fig.layout[key] for key in subplot.layout_keys]
    fig.update_layout(
        {
            scene: dict(
                domain=dict(x=xaxis.domain, y=yaxis.domain),
                camera=dict(projection=dict(type="orthographic")),
            )
        }
    )
    trace.scene = scene
    fig.add_trace(trace)


def add_curve(fig, array, **kwargs):
    array = np.atleast_2d(array)

    kwargs == sanitize_kwargs(kwargs, ["color", "width", "dash"], "line_")

    if len(array) == 1:
        kwargs.update(mode=kwargs.get("mode", "markers"))
    else:
        kwargs.update(mode=kwargs.get("mode", "lines"))

    if np.shape(array)[-1] == 3:
        x, y, z = np.transpose(array)
        row = kwargs.pop("row", None)
        col = kwargs.pop("col", None)
        add_3d_trace(fig, go.Scatter3d(x=x, y=y, z=z, **kwargs), row=row, col=col)

    else:
        x, y = np.transpose(array)
        fig.add_scatter(x=x, y=y, **kwargs)


def add_point(fig, point, size=None, **kwargs):
    if size is not None:
        kwargs["marker_size"] = size
    add_curve(fig, point, mode="markers", **kwargs)


def radial_ticks(fig, step=180, row=1, col=1):
    fig.update_xaxes(tickvals=np.arange(0, 3600, step), row=row, col=col)


def add_title(fig, title, padding=None):
    fig.update_layout(title=title)
    if padding:
        fig.update_layout(margin_t=padding)


def _legend_legacy(self, offset=0):
    self.update_layout(showlegend=True, legend_traceorder="normal")

    # names = list(set([trace['name'] for trace in self._data]))

    visited = []

    for trace in self._data:
        if not trace.get("showlegend", True):
            continue
        if not "name" in trace:
            trace["showlegend"] = False
            continue
        name = trace["name"]
        trace["showlegend"] = (
            visited.count(name) <= offset and visited.count(name) > offset - 1
        )
        visited.append(name)


def _legend(fig, parent=-1, **kwargs):
    """Add a legend with patches / lines style taken from the parent-th plot.
    Call after a given plot with `parent=-1` to add the last plot to the legend.
    Set custom colors, e.g. `fillcolor=black` to overwrite the original color.
    """

    for plot in fig["data"]:
        if not plot.showlegend:
            plot.showlegend = False

    plain = [plot for plot in fig["data"] if not plot.customdata]
    plot = plain[parent]

    args = dict(
        mode=plot.mode, line=plot.line, fill=plot.fill, showlegend=True, name=plot.name
    )
    args.update(kwargs)
    fig.add_scatter(x=[np.nan], y=[np.nan], **args)
    fig["data"][-1].customdata = ["legend"]

    fig.update_layout(showlegend=True)


def square_axes(fig, row=None, col=None):
    if any([isinstance(elem, go.Scatter3d) for elem in fig.data]) or any(
        [isinstance(elem, go.Mesh3d) for elem in fig.data]
    ):
        fig.update_layout(
            scene_aspectratio=dict(x=1, y=1, z=1), scene_aspectmode="data"
        )
    else:
        if row is None and col is None:
            fig.update_yaxes(scaleanchor="x")
            layout = np.array(fig._grid_ref, dtype="object").shape
            nrows, ncols, *inner_dims = layout

            for row in range(1, nrows + 1):
                for col in range(1, ncols + 1):
                    fig.update_xaxes(scaleanchor="y", row=row, col=col)

        fig.update_xaxes(scaleanchor="y", row=row, col=col)


def rasterize(fig, refresh=False, **kwargs):
    try:
        if refresh:
            raise Exception("refresh rasterized image")
        return fig._rasterized
    except:
        img_bytes = fig.to_image(format="png", **kwargs)
        img = Image.open(io.BytesIO(img_bytes))
        img_array = np.array(img)
        fig._rasterized = img_array
        return img_array


def strip_figure(fig, x=True, y=True, row=None, col=None):
    fig.update_layout(template="plotly_white")
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)

    scene_dict = dict(
        xaxis=dict(showgrid=False, zeroline=False, title="", showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, title="", showticklabels=False),
        zaxis=dict(showgrid=False, zeroline=False, title="", showticklabels=False),
    )

    # fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)',
    #                   plot_bgcolor='rgba(0, 0, 0, 0)')
    i = 0  # orthographic projection for all scenes
    while True:
        try:
            scene = fig.layout[f"scene{i if i else ''}"]
            scene["xaxis"] = scene_dict["xaxis"]
            scene["yaxis"] = scene_dict["yaxis"]
            scene["zaxis"] = scene_dict["zaxis"]
            i += 1
        except:
            break

    if x:
        fig.update_xaxes(
            showgrid=False, zeroline=False, showticklabels=False, row=row, col=col
        )
    if y:
        fig.update_yaxes(
            showgrid=False, zeroline=False, showticklabels=False, row=row, col=col
        )

    if x and y:
        top_margin = fig.layout.margin.t if fig.layout.annotations else 0
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=top_margin))


def add_titles(fig, axis, *args, skip=None, repeat=1, x=None, y=None, align=False):
    args = args * repeat
    if skip:
        args = ("",) * skip + args
    if axis == "xy":
        fig._xtitles(args[0])
        fig._ytitles(args[1])

    layout = np.array(fig._grid_ref, dtype="object").shape
    nrows, ncols, *inner_dims = layout

    if x and y:
        if align:
            align = "!_____!"
        else:
            align = ""

        if axis == "x":
            for row in range(nrows):
                for col in range(ncols):
                    fig.update_xaxes(title=align, row=row + 1, col=col + 1)
            fig.add_annotation(
                text=args[0],
                x=x,
                y=y,
                showarrow=False,
                xref="paper",
                yref="paper",
                font_size=14,
            )
        else:
            for row in range(nrows):
                for col in range(ncols):
                    fig.update_yaxes(title=align, row=row + 1, col=col + 1)
            fig.add_annotation(
                text=args[0],
                x=x,
                y=y,
                showarrow=False,
                xref="paper",
                yref="paper",
                font_size=14,
                textangle=-90,
            )

    else:
        for n, title in enumerate(args):
            row = (n // ncols) + 1
            col = (n % ncols) + 1

            if axis == "x":
                fig.update_xaxes(title=title, row=row, col=col)
            if axis == "y":
                fig.update_yaxes(title=title, row=row, col=col)


def log_axis(fig, axes=[], **kwargs):
    row = kwargs.get("row", None)
    col = kwargs.get("col", None)

    if "x" in axes:
        fig.update_xaxes(type="log", row=row, col=col)
    if "y" in axes:
        fig.update_yaxes(type="log", row=row, col=col)


def loop(fig):
    layout = np.array(fig._grid_ref, dtype="object").shape
    nrows, ncols, *inner_dims = layout

    return [
        dict(row=r, col=c)
        for r, c in itertools.product(range(1, nrows + 1), range(1, ncols + 1))
    ]


def blend_scatter(fig, n_segments=25):
    """Remove the last scatter, and reattach it in segments,
    so that they can accumulate opacity.
    """

    data = fig._data.pop()

    x = data["x"]
    y = data["y"]

    segment_length = len(x) // n_segments

    for i in range(0, len(x) * 2, segment_length):
        segment = data.copy()
        x_s = x[i : i + segment_length + 1]
        y_s = y[i : i + segment_length + 1]

        segment["x"] = x_s
        segment["y"] = y_s
        fig.add_trace(segment)


def add_fit(
    fig, degree=1, n=2, type="polynomial", p0=[1, 1], row=None, col=None, **kwargs
):
    """Add a polynomial fit to the data plotted in fig.
    For subplots, you might use `row` and `col`. In this case,
    takes the first data added to the corresponding axis.
    If `row` and `col` are not provided, not provided, take
    the last data added to `fig`.
    """

    if n == 2 and (degree > 1 or type != "polynomial"):
        n = 500

    layout = np.array(fig._grid_ref, dtype="object").shape
    if layout:
        nrows, ncols, *inner_dims = layout

    if row is None and col is None:
        plot_index = -1
        if layout:
            axis = fig._data[-1]["x"].split("x")[-1]
            if not axis:
                row, col = 1, 1
            else:
                row = ((int(axis) - 1) // ncols) + 1
                col = ((int(axis) - 1) % ncols) + 1
    else:
        if row is None:
            row = 1
        if col is None:
            col = 1

        if not layout:
            plot_index = -1
        else:
            plot_index = (row - 1) * nrows + (col - 1) + 1
            if plot_index == 1:
                plot_index = ""

            plot_index = [d["x"] for d in fig._data].index(f"x{plot_index}")

    x, y = [fig._data[plot_index][a] for a in ("x", "y")]

    if type in ("polynomial", "pol"):
        coeffs = np.polyfit(x, y, 1)
        x = np.linspace(min(x), max(x), n)
        y = np.poly1d(coeffs)(x)

    elif type in ("exponential", "exp"):
        popt, cov = scipy.optimize.curve_fit(tools.signal.exponential, x, y, p0)
        x = np.linspace(min(x), max(x), n)
        y = tools.signal.exponential(x, *popt)

    elif type in ("logarithmic", "log"):
        popt, cov = scipy.optimize.curve_fit(tools.signal.log, x, y, p0)
        x = np.linspace(min(x), max(x), n)
        y = tools.signal.log(x, *popt)

    else:
        raise ValueError(f"Unknown model: {type}")

    if not "line_color" in kwargs:
        kwargs["line_color"] = "#EF553B"
    fig.add_scatter(
        x=x,
        y=y,
        mode="lines",
        line_width=2,
        showlegend=False,
        row=row,
        col=col,
        **kwargs,
    )


def highlight_segment(self, test_function):
    """Highlight x-values that obey `test_function`(x) == True,
    and plot the remaining values with reduced opacity.
    This works on the last added curve of `self`.
    """

    data = self._data.pop()

    # If we have no line_color, we'll colorize them consecutively.
    # At the moment, this fails if some line properties are provided
    # but no line_color. If this ever happens, we have to differentiate
    # these cases in more detail.

    if not "line" in data.keys():
        count = max([0] + [f.get("count", 0) for f in self._data]) + 1
        data["count"] = count
        data["line"] = dict(color=colors[count - 1])

    x = data["x"]
    y = data["y"]

    edges = np.where(np.diff(test_function(x)) != 0)[0]
    edges = np.concatenate([[0], edges, [0]])

    for left, right in zip(edges, edges[1:]):
        segment = data.copy()
        segment["x"] = x[left : right - 1]
        segment["y"] = y[left : right - 1]

        if len(segment["x"]) < 2:
            continue

        if test_function(np.mean(segment["x"])):
            opacity = 1
        else:
            opacity = 0.25

        segment["opacity"] = opacity

        self._data.append(segment)


def colorize(iterable, continuous=True, key=None, sort=True, enum=False, **kwargs):
    """Returns an iterator that contains a (color, element) pair for each
    element in `iterable`.
    If `continuous` is True, the colors are laid out along a continuous
    color scale.
    If no key is provided, this mapping is done using the index of each
    element in the iterable. Else, a custom function can be provided,
    which should accept `iterable` and and an element of `iterable` as
    its arguments.
    """

    if not key:
        iterable = list(iterable)

        def key(i, x):
            return [id(elem) for elem in i].index(id(x))

    if sort:
        iterable = sorted(iterable, key=lambda x: key(iterable, x))

    if enum:
        if continuous:
            vals = [key(iterable, elem) for elem in iterable]
            color_list = [
                scaled_color(key(iterable, elem), min(vals), max(vals), **kwargs)
                for elem in iterable
            ]
            return zip(range(len(iterable)), color_list, iterable)
        else:
            return [
                (i, colors[i % len(colors)], elem) for i, elem in enumerate(iterable)
            ]

    else:
        if continuous:
            vals = [key(iterable, elem) for elem in iterable]
            color_list = [
                scaled_color(key(iterable, elem), min(vals), max(vals), **kwargs)
                for elem in iterable
            ]
            return zip(color_list, iterable)
        else:
            return [(colors[i % len(colors)], elem) for i, elem in enumerate(iterable)]


def neat(values):
    """Returns `values`, but with all points exchanged with np.nan after
    the first occurence of a large jump.
    """

    index = np.where(np.abs(np.diff(values)) > np.mean(values) / 5)[0]
    if index.size:
        selection = np.ones_like(values)
        selection[index[0] :] = 0
        return values[selection.astype(bool)]
    else:
        return values


def scaled_color(
    value,
    low=0,
    high=1,
    diverging=False,
    narrow=False,
    alpha=False,
    return_string=True,
    colorscale="YlGnBu",
):
    """Return a color whose value represents its position between low and high.
    alpha = 'min' or 'max' returns additionaly an opacity parameter.

    Provide an iterable object to `low` in order to set low, high = min(low), max(low).
    """

    if hasattr(value, "__iter__"):
        return [
            scaled_color(
                v, value, high, diverging, narrow, alpha, return_string, colorscale
            )
            for v in value
        ]

    if hasattr(low, "__iter__"):
        if len(low) >= 2:
            low, high = min(low), max(low)

    if value > high:
        value = high
    if value < low:
        value = low

    if diverging:
        if colorscale == "YlGnBu":
            colorscale = "Spectral"
        try:
            colorscale = getattr(px.colors.diverging, colorscale)
        except AttributeError:
            colorscale = getattr(px.colors.sequential, colorscale)
        except:
            raise ValueError(
                f'"{colorscale}" not found in px.colors.diverging or px.colors.sequential.'
            )
    else:
        colorscale = getattr(px.colors.sequential, colorscale)

    if "#" in colorscale[0]:
        colorscale = cl.to_rgb(
            [
                tuple(int((c).lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
                for c in colorscale
            ]
        )

    color_list = cl.to_numeric(cl.to_rgb(cl.interp(colorscale, 101)))[::-1]

    if narrow:
        value = 25 + 50 * (value - low) / ((high) - low)
    else:
        value = 100 * (value - low) / ((high) - low)
    color = color_list[int(np.floor(value))]

    if alpha == "max":
        opacity = value / 100
    elif alpha == "min":
        opacity = 1 - value / 100

    if return_string:
        color = Color(tuple(int(c) for c in color))
        if alpha:
            return color, opacity
        else:
            return color
    else:
        if alpha:
            return np.array(color) / 255, opacity
        else:
            return np.array(color) / 255





def subpos(nrows=1, ncols=1, position=None, flip=False):
    """Return a dict containing the  coordinates of the i-th subplot with `nrow`
    rows and `ncol` columns.
    Feed this directly into a plotly trace, e.g., using something in the line of
    fig.add_scatter(x=x, y=y, **subpos(fig, i)).
    Alternatively, the first argument can be of type <go.Figure>, in which case
    row an column numbers are inferred from this, and `ncols` does not have to
    be provided.
    """

    if isinstance(nrows, go.Figure):
        if position is None:
            position = ncols
        nrows, ncols, *inner_dims = np.array(nrows._grid_ref, dtype="object").shape

    # not very elegant, but this rearanges arguments for backwards compatibility
    if flip:
        row, col = (np.concatenate(np.indices((nrows, ncols)).T) + 1)[position]
        return dict(row=int(row), col=int(col))
    else:
        row, col = (np.concatenate(np.indices((ncols, nrows)).T) + 1)[position]
        return dict(row=int(col), col=int(row))


def ijk(N1, N2):
    """Return the indices of triangles of a regular grid of `N1` x `N2`
    points. The three return values correspond to the i, j and k parameter
    in plotly's Mesh3d function.
    """

    ijk = [
        (o + i, o + i + 1, o + i + N1)
        for i in range(N1 - 1)
        for o in range(0, (N1) * (N2 - 1), N1)
    ]
    ijk += [
        (o + i + N1, o + i + 1, o + i + N1 + 1)
        for i in range(N1 - 1)
        for o in range(0, (N1) * (N2 - 1), N1)
    ]

    return zip(*ijk)


def make_arrow(x, y, s=0.125):
    """Given a line of two points, return a multi-line segment
    that is an arrow.
    This is a workaround to easily draw arrows in plotly and
    matplotlib.
    """

    (x1, x2), (y1, y2) = x, y
    s = 1 / s
    x = [
        x1,
        x2,
        x2 - (y2 - y1) / s,
        x2 + (x2 - x1) / s * np.sqrt(2),
        x2 + (y2 - y1) / s,
        x2,
    ]
    y = [
        y1,
        y2,
        y2 + (x2 - x1) / s,
        y2 + (y2 - y1) / s * np.sqrt(2),
        y2 - (x2 - x1) / s,
        y2,
    ]

    return x, y


small = dict(width=500, height=300, margin=dict(l=10, r=10, t=20, b=10))
tiny = dict(width=300, height=200, margin=dict(l=10, r=10, t=20, b=10))
small_square = dict(width=400, height=400, margin=dict(l=10, r=10, t=20, b=10))
square = dict(width=600, height=600, margin=dict(l=10, r=10, t=20, b=10))
tall = dict(width=600, height=800, margin=dict(l=10, r=10, t=40, b=10))
large = dict(width=1000, height=800, margin=dict(l=10, r=10, t=20, b=10))
regular = dict(width=700, height=450, margin=dict(l=10, r=10, t=40, b=10))
short = dict(width=700, height=300, margin=dict(l=10, r=10, t=40, b=10))
wide = dict(width=1000, height=450, margin=dict(l=10, r=10, t=40, b=10))

portrait = dict(width=450, height=700, margin=dict(l=10, r=10, t=40, b=10))
portrait_small = dict(width=300, height=500, margin=dict(l=10, r=10, t=20, b=10))
