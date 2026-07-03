from functools import wraps
from numbers import Number
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

# set matplotlib style
from cycler import cycler
from IPython import get_ipython
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter, writers
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.gridspec import GridSpec

from phipy_plots.color import colors

plt.style.use("default")
plt.rcParams["axes.prop_cycle"] = cycler(color=colors)
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.labelsize"] = 9
mpl.rcParams["xtick.labelsize"] = 8
mpl.rcParams["ytick.labelsize"] = 8
mpl.rcParams["font.family"] = "Helvetica Neue"  #'sans-serif'
mpl.rcParams["grid.color"] = "#CCC"
mpl.rcParams["axes.edgecolor"] = "#CCC"
mpl.rcParams["axes.linewidth"] = 0.5
mpl.rcParams["xtick.color"] = "#CCC"
mpl.rcParams["ytick.color"] = "#CCC"
mpl.rcParams["xtick.labelcolor"] = "#444"
mpl.rcParams["ytick.labelcolor"] = "#444"
mpl.rcParams["xtick.major.width"] = 0.5
mpl.rcParams["ytick.major.width"] = 0.5
mpl.rcParams["xtick.minor.width"] = 0.5
mpl.rcParams["ytick.minor.width"] = 0.5
mpl.rcParams["legend.edgecolor"] = "#CCC"
mpl.rcParams["legend.borderpad"] = 0.5
mpl.rcParams["legend.borderaxespad"] = 0.5
mpl.rcParams["legend.framealpha"] = 0.8
mpl.rcParams["legend.frameon"] = False
mpl.rcParams["legend.fancybox"] = False
mpl.rcParams["axes.titlesize"] = 10
mpl.rcParams["legend.fontsize"] = 8
mpl.rcParams["figure.labelsize"] = 9


def add_delta_bracket(
    ax,
    y: Tuple[Number, Number] = [0, 1],
    unit: str = "",
    x_shift: Number = -0.03,
    pad: Number = 0,
    **kwargs,
):
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

    ymin, ymax = y

    ax.annotate(
        "",
        xy=(x_shift, ymax),
        xycoords=trans,
        xytext=(x_shift, ymin),
        textcoords=trans,
        arrowprops=dict(
            arrowstyle="-",
            connectionstyle="bar",
            linewidth=mpl.rcParams["axes.linewidth"],
            color=mpl.rcParams["axes.edgecolor"],
            **kwargs,
        ),
        clip_on=False,
    )

    scale = 1
    if unit == "%":
        scale = 100

    ax.annotate(
        f"{int((ymax - ymin)) * scale} {unit}",
        xy=(x_shift, (ymin + ymax) / 2),
        xycoords=trans,
        xytext=(x_shift + pad, (ymin + ymax) / 2),
        textcoords=trans,
        va="center",
        ha="right",
        clip_on=False,
        fontsize=mpl.rcParams["ytick.labelsize"],
    )


def set_categorical_x(
    ax: mpl.axes.Axes, bounds: Optional[Tuple[Number, Number]] = None
):
    ax.tick_params(axis="x", which="both", bottom=False, top=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    if bounds is None:
        bounds = ax.get_xlim()
    ax.spines.left.set_bounds(*bounds)


def set_categorical_y(
    ax: mpl.axes.Axes, bounds: Optional[Tuple[Number, Number]] = None
):
    ax.tick_params(axis="y", which="both", left=False, right=False)

    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # if bounds is None:
    #     bounds = ax.get_ylim()
    # ax.spines.bottom.set_bounds(*bounds)
    return ax


def only_x_spine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="y", which="both", left=False, top=False)
    ax.spines.bottom.set_bounds(*ax.get_xlim())

    return ax


def no_x_axis(ax):
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    return ax


def select_points_on_figure(fig, n=5, timeout=0):
    """
    This wrapper is mainly here to document how to set that up.

    The venve needs the python packages `ipympl` and `PyQt6` installed.

    ```
    matplotlib.use('QtAgg')
    %matplotlib widget
    ```

    Then, before creating the figure set up to use QT:
        `%matplotlib qt`
    and after selecting the points reset:
        `%matplotlib widget`

    """

    get_ipython().run_line_magic("matplotlib", "qt")

    plt.figure(fig.number)
    pts = plt.ginput(n=n, timeout=timeout)

    get_ipython().run_line_magic("matplotlib", "widget")

    return np.array(pts, dtype=float)


def point_selector(n=5, timeout=0, backend_in="qt", backend_out="widget"):
    """
    Decorator for a function that creates and returns a matplotlib figure.

    The decorated function returns the selected points as an (N, 2) numpy array.
    """

    def decorator(make_figure_func):
        @wraps(make_figure_func)
        def wrapper(*args, **kwargs):
            ip = get_ipython()

            ip.run_line_magic("matplotlib", backend_in)
            fig = make_figure_func(*args, **kwargs)

            plt.figure(fig.number)
            pts = plt.ginput(n=n, timeout=timeout)

            plt.close(fig)
            ip.run_line_magic("matplotlib", backend_out)

            return np.array(pts, dtype=float)

        return wrapper

    return decorator


def colorbar_with_labels(mappable, labels, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    cbar = plt.colorbar(mappable, ax=ax, **kwargs)
    n = len(labels)
    vmin, vmax = cbar.mappable.get_clim()
    centers = np.linspace(vmin, vmax, 2 * n + 1)[1::2]
    cbar.set_ticks(centers)
    cbar.ax.set_yticklabels(labels)
    return cbar


def animate_imshow(
    frames,
    ax=None,
    interval=100,
    title="Frame {i}",
    axis="off",
    blit=True,
    repeat=True,
    as_html=False,
    **imshow_kwargs,
):
    from IPython.display import HTML

    frames = list(frames)
    if not frames:
        raise ValueError("Need at least one frame.")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    imshow_kwargs.setdefault("animated", blit)
    image = ax.imshow(frames[0], **imshow_kwargs)

    if axis is not None:
        ax.axis(axis)

    def set_title(i):
        if title is None:
            return []
        if callable(title):
            text = title(i, frames[i])
        else:
            text = title.format(i=i, frame=frames[i])
        ax.set_title(text)
        return [ax.title]

    set_title(0)

    def update(i):
        image.set_array(frames[i])
        return [image] + set_title(i)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval,
        blit=blit,
        repeat=repeat,
    )
    animation.fig = fig
    animation.ax = ax
    animation.image = image

    if as_html:
        html = HTML(animation.to_jshtml())
        html.animation = animation
        return html

    return animation


class Animator:
    def __init__(self, dpi=None, background="white", close=True):
        self.frames = []
        self.dpi = dpi
        self.background = background
        self.close = close
        self.shape = None

    def __len__(self):
        return len(self.frames)

    def clear(self):
        self.frames.clear()
        self.shape = None

    def add_frame(self, fig=None, dpi=None, close=None):
        if fig is None:
            fig = plt.gcf()
        if dpi is None:
            dpi = self.dpi
        if close is None:
            close = self.close

        original_dpi = fig.dpi
        try:
            if dpi is not None:
                fig.set_dpi(dpi)

            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            frame = np.asarray(canvas.buffer_rgba()).copy()

            if self.background is not None:
                frame = self._composite_background(frame)

            if self.shape is None:
                self.shape = frame.shape
            elif frame.shape != self.shape:
                raise ValueError(
                    f"Frame shape {frame.shape} does not match first frame {self.shape}."
                )

            self.frames.append(frame)
        finally:
            if dpi is not None:
                fig.set_dpi(original_dpi)
            if close:
                plt.close(fig)

        return frame

    def save(self, path, framerate=25, dpi=None, **kwargs):
        if not self.frames:
            raise ValueError("No frames to save.")

        path = str(path)
        if dpi is None:
            dpi = self.dpi or 100

        height, width = self.frames[0].shape[:2]
        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis("off")
        image = ax.imshow(self.frames[0], animated=True)

        def update(i):
            image.set_array(self.frames[i])
            return [image]

        animation = FuncAnimation(
            fig,
            update,
            frames=len(self.frames),
            interval=1000 / framerate,
            blit=True,
        )

        suffix = path.rsplit(".", 1)[-1].lower()
        if suffix == "gif":
            writer = PillowWriter(fps=framerate)
        elif suffix in {"mp4", "m4v", "mov"}:
            if not writers.is_available("ffmpeg"):
                plt.close(fig)
                raise RuntimeError("Saving MP4 requires ffmpeg to be installed.")
            writer = FFMpegWriter(fps=framerate)
        else:
            writer = kwargs.pop("writer", None)

        animation.save(path, writer=writer, dpi=dpi, **kwargs)
        plt.close(fig)
        return animation

    def _composite_background(self, frame):
        if frame.shape[-1] != 4:
            return frame

        rgb = frame[..., :3].astype(float)
        alpha = frame[..., 3:4].astype(float) / 255
        background = np.array(mpl.colors.to_rgb(self.background)) * 255
        return (rgb * alpha + background * (1 - alpha)).round().astype(np.uint8)


def scale_text(fig, factor):
    for ax in fig.axes:
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + list(ax.get_xticklabels())
            + list(ax.get_yticklabels())
            + list(ax.texts)
        ):
            try:
                item.set_fontsize(item.get_fontsize() * factor)
            except AttributeError:
                pass

        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(text.get_fontsize() * factor)

    suptitle = fig._suptitle
    if suptitle is not None:
        suptitle.set_fontsize(suptitle.get_fontsize() * factor)
