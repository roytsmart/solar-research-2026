"""
The O V scene ESIS recovered from its own flight images.
"""

import pathlib
import matplotlib.animation
import matplotlib.artist
import matplotlib.axes
import matplotlib.cm
import matplotlib.lines
import matplotlib.patches
import matplotlib.patheffects
import matplotlib.colors
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import astropy.units as u
import astropy.time
import astropy.coordinates
import astropy.constants
import astropy.visualization
import named_arrays as na
import sdo
import esis
from .._temperature import temperature_line, order_temperature
from .._gaussians import fit_gaussians, seed_gaussians
from ._color import velocity_color_default, percentile_default
from ._layout import figsize_default, bottom_default, height_default
from ._path import default_path

__all__ = [
    "path_level_4_default",
    "level_4",
    "level_4_velocity",
    "level_4_lines",
    "level_4_event",
    "level_4_event_gaussians",
    "level_4_event_contours",
    "level_4_event_history",
    "level_4_event_motion",
]

#: Where the Level-4 product of the O V line is kept.
#:
#: The published copy lives on a share, and reading it from there takes an
#: hour where reading it from a local disk takes five seconds, so a local
#: copy is used if one has been made and the share is the fallback.
path_level_4_default = pathlib.Path.home() / ".spd2026/data/esis_level_4_o_v_630.fits"

#: Where the published copy lives, if there is no local one.
path_level_4_remote = pathlib.Path(
    "Z:/esis_level4_gpu/fits/esis_level_4_o_v_630.fits",
)

#: The position of the image of the sky, as a fraction of the figure.
#:
#: Wider than the rect the other figures use, since this is the whole field
#: rather than the patch the raster covers, and it is nearly square: at the
#: height of a slide a square image is as wide as it is tall, and this rect
#: is only there to leave it room and to keep it clear of the key.
#: The gap between the key and the image has to hold four things: the ticks
#: and label of the velocity scale on the right of the key, and the ticks and
#: label of the vertical axis of the image.
_rect_image = (0.225, bottom_default, 0.52, height_default)
_rect_key = (0.065, bottom_default, 0.024, height_default)

#: The power of ten taken out of the brightness scale of the key, so that its
#: two tick labels are short numbers rather than a pair of digits with an
#: exponent hung underneath them, on top of the unit. In erg the brightness
#: runs to a few tens and needs nothing taken out of it.
_scale_key = 1

#: Where event E is, the compact bidirectional event which has been studied
#: so far, as recorded in the README distributed with the Level-4 product.
position_event_e = na.Cartesian2dVectorArray(
    x=47.8 * u.arcsec,
    y=-87.8 * u.arcsec,
)

#: How far north of event E the close-up figures look, and how wide.
#:
#: The event does not stay where it was found, so the region they show is
#: not centered on it. Written down here so that the box drawn on the whole
#: field is the region those figures actually show rather than a second
#: guess at it.
offset_event_default = 19 * u.arcsec
radius_event_default = 29 * u.arcsec

#: The middle of the region the close-up figures show.
center_event_default = na.Cartesian2dVectorArray(
    x=position_event_e.x,
    y=position_event_e.y + offset_event_default,
)

#: Where the second event is, near the south of the field: a compact
#: brightening with blue and red Doppler shifts side by side, found by eye in
#: the O V cube for the group-meeting version of the talk. It lasts from
#: about 18:07 to 18:10 and hardly moves while it does, so unlike event E the
#: close-up figures are centered on it, with the same half width.
position_event_south = na.Cartesian2dVectorArray(
    x=47 * u.arcsec,
    y=-299 * u.arcsec,
)


def _open(path_data: None | pathlib.Path) -> "esis.data.Level_4":
    """Read the Level-4 product, preferring a local copy to the share."""
    if path_data is None:
        path_data = path_level_4_default
        if not path_data.exists():  # pragma: nocover
            path_data = path_level_4_remote
    return esis.data.Level_4.from_fits(path_data)


def level_4(
    path_data: None | pathlib.Path = None,
    center: None | na.Cartesian2dVectorArray = None,
    radius: u.Quantity = radius_event_default,
    center_box: None | na.Cartesian2dVectorArray = center_event_default,
    radius_box: u.Quantity = radius_event_default,
    color_box: str = "red",
    linewidth_box: float = 2.5,
    unit_intensity: u.UnitBase = u.erg / (u.AA * u.s * u.cm**2 * u.deg**2),
    index_time_reference: int = 15,
    velocity_color: u.Quantity = velocity_color_default,
    percentile: float = percentile_default,
    figsize: tuple[float, float] = figsize_default,
    dpi: float = 150,
    fps: int = 5,
    timestamp: bool = True,
    suffix: str = ".mp4",
    path: None | pathlib.Path = None,
) -> pathlib.Path:
    r"""
    Loop over the :math:`\text{O\,V}\;630\,\AA` scene ESIS recovered.

    This is the flight, not a simulation of it: the Level-4 product is the
    time-dependent MART inversion of the thirty Level-1 exposures, on a grid
    of three quarters of an arcsecond and seventeen and a half kilometers per
    second, registered to the sky so that a feature fixed on the Sun stays
    put from one frame to the next.

    The scene is drawn as a false-color image, where the hue is the Doppler
    velocity, on the same color scale as the figures of the synthetic scene.
    A color therefore means the same thing here as it does there, and what
    ESIS actually recovered can be held against what it was expected to.

    The whole field is shown, which is some fourteen arcminutes across,
    where the figures built from the IRIS raster cover a couple of
    arcminutes of it.

    Parameters
    ----------
    path_data
        The Level-4 file to read.
        If :obj:`None`, a local copy if one exists and the published copy on
        the share otherwise, see :obj:`path_level_4_default`.
    center
        The place on the sky to look at.
        If :obj:`None`, the whole field. :obj:`position_event_e` is where
        the event studied so far sits.
    radius
        The half width of the region shown when `center` is given.
    center_box
        The middle of a region to outline on the image.

        The region the close-up figures look at, so that a reader can see
        where in the field the event they are about sits. If :obj:`None`,
        no box is drawn, which is what a figure already cropped to that
        region wants.
    radius_box
        The half width of the outlined region.
    color_box
        The color to outline it in. Outlined in black underneath, so that it
        is visible against the bright parts of the image as well as the dark.
    linewidth_box
        The width of the outline.
    unit_intensity
        The unit of the brightness axis of the key, see
        :func:`_energy_photon`. Per angstrom, since this figure shows the
        spectrum rather than the light of the line added up.
    index_time_reference
        The frame whose brightness sets the color scale of every frame.
        The default is the frame the distortion fit was optimized against,
        which is also near the middle of the flight, where the rocket is
        highest and the signal is strongest. The frames at either end are
        therefore dimmer, as they were.
    velocity_color
        The Doppler velocity mapped to each end of the visible spectrum,
        which is the range the other figures use.
    percentile
        The percentile of the reference frame placed at the top of the
        brightness scale, separately at each wavelength.
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution of the saved animation in dots per inch.
    fps
        The number of frames per second in the saved animation.
    timestamp
        Whether to write the time of each exposure above the image.
    suffix
        The file type of the animation, either ``".mp4"`` or ``".gif"``.
        Ignored if `path` is given.
    path
        The location to save the animation.
        If :obj:`None`, it is saved alongside the other figures.

    Notes
    -----
    The radiometry of the four channels is relative rather than absolute,
    so the structure and the velocities of this scene are to be trusted
    further than its brightness.
    """
    a = _open(path_data)

    axis_time = a.axis_time
    axis_wavelength = a.axis_wavelength
    axis_xy = (a.axis_x, a.axis_y)

    position = a.inputs.position
    wavelength = a.inputs.wavelength
    wavelength_rest = a.wavelength_center.ndarray[0]

    # The product is given in wavelength, while the color scale of the other
    # figures is given in Doppler velocity, so the axis is expressed the way
    # the color scale expects to be told about it.
    equivalency = u.doppler_optical(wavelength_rest)
    velocity = wavelength.to(u.km / u.s, equivalencies=equivalency)

    unit_wavelength = na.unit(wavelength)
    unit_position = na.unit(position.x)

    # One scale for the whole flight, taken from the brightest part of it,
    # so that the rise and fall of the signal as the rocket climbs out of
    # the atmosphere and falls back into it is left in rather than divided
    # out.
    reference = a.outputs[{axis_time: index_time_reference}]
    vmax = np.nanpercentile(reference, percentile, axis=axis_xy)

    time = a.inputs.time

    if path is None:
        stem = "level-4-o-v"
        if center is not None:
            stem = f"{stem}-event"
        if center_box is not None:
            stem = f"{stem}-box"
        path = default_path / f"{stem}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)

    with astropy.visualization.quantity_support():

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(_rect_image)
        cax = fig.add_axes(_rect_key)
        cax_twin = cax.twinx()

        if center_box is not None:
            ax.add_patch(
                matplotlib.patches.Rectangle(
                    xy=(
                        (center_box.x - radius_box).value,
                        (center_box.y - radius_box).value,
                    ),
                    width=(2 * radius_box).value,
                    height=(2 * radius_box).value,
                    fill=False,
                    edgecolor=color_box,
                    linewidth=linewidth_box,
                    # Above the image, which is drawn after this and would
                    # otherwise cover it.
                    zorder=10,
                    path_effects=[
                        matplotlib.patheffects.withStroke(
                            linewidth=linewidth_box + 1.6,
                            foreground="black",
                        )
                    ],
                )
            )

        if timestamp:
            text = fig.text(
                x=0.5,
                y=0.985,
                s="",
                ha="center",
                va="top",
            )
        else:
            text = None

        if path.suffix == ".gif":
            writer = matplotlib.animation.PillowWriter(fps=fps)
        else:
            writer = matplotlib.animation.FFMpegWriter(
                fps=fps,
                codec="h264",
                extra_args=["-pix_fmt", "yuv420p", "-crf", "14"],
            )

        drawn = [-1]

        def func(index: int) -> list[matplotlib.artist.Artist]:

            if index == drawn[0]:
                return [*ax.collections]
            first = drawn[0] < 0
            drawn[0] = index

            for artist in ax.collections:
                artist.remove()

            colorbar = na.plt.rgbmesh(
                velocity,
                position.x,
                position.y,
                C=a.outputs[{axis_time: index}],
                axis_wavelength=axis_wavelength,
                ax=ax,
                vmin=0,
                vmax=vmax,
                wavelength_min=-velocity_color,
                wavelength_max=+velocity_color,
            )

            # The scale never changes, so the key is only drawn once.
            if first:
                brightness = (colorbar.inputs.x * _energy_photon(a, 0)).to(
                    unit_intensity
                ) / (unit_intensity * _scale_key)

                na.plt.pcolormesh(
                    brightness,
                    colorbar.inputs.y.to(
                        unit_wavelength,
                        equivalencies=equivalency,
                    ),
                    C=colorbar.outputs,
                    axis_rgb=axis_wavelength,
                    ax=cax,
                )
                na.plt.pcolormesh(
                    brightness,
                    colorbar.inputs.y,
                    C=colorbar.outputs,
                    axis_rgb=axis_wavelength,
                    ax=cax_twin,
                )

                cax.set_ylabel(f"wavelength ({unit_wavelength:latex_inline})")
                cax.set_ylim(
                    (-velocity_color).to(unit_wavelength, equivalencies=equivalency),
                    (+velocity_color).to(unit_wavelength, equivalencies=equivalency),
                )
                cax.set_xlim(0, brightness.max().ndarray)
                label = f"{unit_intensity:latex_inline}"
                if _scale_key != 1:  # pragma: nocover
                    label = f"$10^{{{int(np.log10(_scale_key))}}}$ {label}"
                cax.set_xlabel(label)
                cax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2))
                cax.tick_params(axis="x", labelrotation=45, labelsize="small")

                cax_twin.set_ylabel(f"velocity ({velocity_color.unit:latex_inline})")
                cax_twin.set_ylim(-velocity_color, +velocity_color)

                for artist in [*cax.collections, *cax_twin.collections]:
                    artist.set_rasterized(True)

            for artist in ax.collections:
                artist.set_rasterized(True)

            # A crop is a change of limits rather than of the data: every
            # cell is drawn and the ones outside are clipped, so the region
            # shown is drawn at the resolution of the axes.
            if center is None:
                ax.set_xlim(position.x.min().ndarray, position.x.max().ndarray)
                ax.set_ylim(position.y.min().ndarray, position.y.max().ndarray)
            else:
                ax.set_xlim((center.x - radius).value, (center.x + radius).value)
                ax.set_ylim((center.y - radius).value, (center.y + radius).value)
            ax.set_aspect("equal")
            ax.set_xlabel(f"helioprojective $x$ ({unit_position:latex_inline})")
            ax.set_ylabel(f"helioprojective $y$ ({unit_position:latex_inline})")

            if text is not None:
                t = time[{axis_time: index}].ndarray
                text.set_text(t.strftime("%Y-%m-%d %H:%M:%S UTC"))

            return [*ax.collections]

        ani = matplotlib.animation.FuncAnimation(
            fig=fig,
            func=func,
            frames=a.outputs.shape[axis_time],
        )

        ani.save(
            filename=path,
            writer=writer,
            dpi=dpi,
        )

    plt.close(fig)

    return path


def level_4_velocity(
    path_data: None | pathlib.Path = None,
    center: None | na.Cartesian2dVectorArray = None,
    radius: u.Quantity = radius_event_default,
    velocity_limit: u.Quantity = 80 * u.km / u.s,
    cmap: str = "RdBu_r",
    color_bad: str = "black",
    figsize: tuple[float, float] = figsize_default,
    dpi: float = 150,
    fps: int = 5,
    timestamp: bool = True,
    suffix: str = ".mp4",
    path: None | pathlib.Path = None,
) -> pathlib.Path:
    r"""
    Loop over the Doppler shift ESIS recovered.

    The same flight as :func:`level_4`, reduced from a spectrum at every
    place on the sky to a single number: the median of the line profile,
    which is the velocity that half the light of that pixel is moving faster
    than and half slower.

    Drawn the way a Doppler shift is conventionally drawn: blue where the
    plasma is coming toward us, red where it is going away, and white where
    it is at rest. The two signs then separate at a glance rather than
    having to be read off a ramp, and, since the quiet Sun is near rest, the
    background settles into the pale middle of the colormap instead of into
    a busy mid-tone.

    The range is a great deal narrower than the range of the spectrum: the
    line moves by a few kilometers per second almost everywhere, so a scale
    reaching the ends of the recovered spectrum would leave the whole field
    the same shade of gray.

    Parameters
    ----------
    path_data
        The Level-4 file to read.
        If :obj:`None`, a local copy if one exists and the published copy on
        the share otherwise, see :obj:`path_level_4_default`.
    center
        The place on the sky to look at.
        If :obj:`None`, the whole field. :obj:`position_event_e` is where
        the event studied so far sits.
    radius
        The half width of the region shown when `center` is given.
    velocity_limit
        The Doppler velocity at each end of the colormap.
    cmap
        The colormap.
        One which runs through white at its middle, since the quantity is
        signed and its middle is where the plasma is at rest.
    color_bad
        The color of the places with no median to report.
        Outside the field stop no light was recorded, so the median of
        nothing is nothing. It has to be a color the colormap itself never
        takes, or those places would read as a velocity: the middle of a
        diverging colormap is white, so white would say the plasma was at
        rest rather than absent.
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution of the saved animation in dots per inch.
    fps
        The number of frames per second in the saved animation.
    timestamp
        Whether to write the time of each exposure above the image.
    suffix
        The file type of the animation, either ``".mp4"`` or ``".gif"``.
        Ignored if `path` is given.
    path
        The location to save the animation.
        If :obj:`None`, it is saved alongside the other figures.

    Notes
    -----
    A median is only as meaningful as the profile it is taken of. Where the
    line is faint the profile is mostly noise and so is its median, which is
    why the faintest places look like static rather than like plasma.
    """
    a = _open(path_data)

    axis_time = a.axis_time
    axis_wavelength = a.axis_wavelength

    position = a.inputs.position
    wavelength_rest = a.wavelength_center.ndarray[0]
    equivalency = u.doppler_optical(wavelength_rest)
    velocity = a.inputs.wavelength.to(u.km / u.s, equivalencies=equivalency)

    unit_position = na.unit(position.x)
    unit_velocity = velocity_limit.unit

    num_time = a.outputs.shape[axis_time]

    # Worked out for every frame before any is drawn, which costs a few
    # hundred megabytes and takes a few seconds, and saves recomputing a
    # frame each time the animation revisits it.
    median = [
        na.pdf.median(
            x=velocity,
            f=a.outputs[{axis_time: i}],
            axis=axis_wavelength,
        ).to_value(unit_velocity)
        for i in range(num_time)
    ]

    time = a.inputs.time

    colormap = plt.get_cmap(cmap).with_extremes(bad=color_bad)

    if path is None:
        stem = "level-4-o-v-velocity"
        if center is not None:
            stem = f"{stem}-event"
        path = default_path / f"{stem}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)

    with astropy.visualization.quantity_support():

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(_rect_image)
        cax = fig.add_axes(_rect_key)

        norm = matplotlib.colors.Normalize(
            vmin=-velocity_limit.to_value(unit_velocity),
            vmax=+velocity_limit.to_value(unit_velocity),
        )
        fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap),
            cax=cax,
            label=f"LOS velocity ({unit_velocity:latex_inline})",
        )
        cax.yaxis.set_ticks_position("left")
        cax.yaxis.set_label_position("left")

        if timestamp:
            text = fig.text(x=0.5, y=0.985, s="", ha="center", va="top")
        else:
            text = None

        if path.suffix == ".gif":
            writer = matplotlib.animation.PillowWriter(fps=fps)
        else:
            writer = matplotlib.animation.FFMpegWriter(
                fps=fps,
                codec="h264",
                extra_args=["-pix_fmt", "yuv420p", "-crf", "14"],
            )

        drawn = [-1]

        def func(index: int) -> list[matplotlib.artist.Artist]:

            if index == drawn[0]:
                return [*ax.collections]
            drawn[0] = index

            for artist in ax.collections:
                artist.remove()

            na.plt.pcolormesh(
                position.x,
                position.y,
                C=median[index],
                ax=ax,
                cmap=colormap,
                norm=norm,
            )

            for artist in ax.collections:
                artist.set_rasterized(True)

            # Set every frame, since removing the last one does not put the
            # limits back and drawing the next only ever widens them.
            #
            # A crop is a change of limits rather than of the data: every
            # cell is drawn and the ones outside are clipped, which costs a
            # little time and means the region shown is drawn at the full
            # resolution of the axes rather than of the whole field.
            if center is None:
                ax.set_xlim(position.x.min().ndarray, position.x.max().ndarray)
                ax.set_ylim(position.y.min().ndarray, position.y.max().ndarray)
            else:
                ax.set_xlim((center.x - radius).value, (center.x + radius).value)
                ax.set_ylim((center.y - radius).value, (center.y + radius).value)
            ax.set_aspect("equal")
            ax.set_xlabel(f"helioprojective $x$ ({unit_position:latex_inline})")
            ax.set_ylabel(f"helioprojective $y$ ({unit_position:latex_inline})")

            if text is not None:
                t = time[{axis_time: index}].ndarray
                text.set_text(t.strftime("%Y-%m-%d %H:%M:%S UTC"))

            return [*ax.collections]

        ani = matplotlib.animation.FuncAnimation(
            fig=fig,
            func=func,
            frames=num_time,
        )

        ani.save(filename=path, writer=writer, dpi=dpi)

    plt.close(fig)

    return path


def _intensity(a: "esis.data.Level_4") -> na.AbstractScalarArray:
    r"""
    The light of each line window, integrated over its Doppler range.

    :attr:`esis.data.Level_4.intensity` adds the cells of a window together
    without multiplying by the width of a cell, so what it returns is still a
    spectral radiance: its unit keeps the :math:`\text{\AA}^{-1}` that an
    integral over wavelength would have cancelled.

    Each window is a grid uniform in Doppler velocity about its own rest
    wavelength, so its cells are all one width, and that width is the only
    thing the sum is missing. The windows do not share it, though, running
    from :math:`0.0341\,\text{\AA}` at He I to :math:`0.0368\,\text{\AA}` at
    O V, so leaving it out is not one constant factor across the lines but
    five different ones, and the lines cannot be held against each other
    until it is put back.

    Parameters
    ----------
    a
        The Level-4 product to integrate.
    """
    axis = a.axis_wavelength

    result = [
        a.outputs[a.window(i)].sum(axis) * _width(a, i) for i in range(a.num_line)
    ]

    return na.stack(result, axis=a.axis_line)


def _width(a: "esis.data.Level_4", index_line: int) -> u.Quantity:
    """
    The width of one cell of a line's spectral window.

    Parameters
    ----------
    a
        The Level-4 product the window belongs to.
    index_line
        The index of the spectral line.
    """
    axis = a.axis_wavelength

    # One more vertex than there are cells, and the extra one is why the
    # window itself cannot be used: it stops at the last cell.
    num = a.num_velocity + 1
    vertices = a.inputs.wavelength[
        {axis: slice(index_line * num, (index_line + 1) * num)}
    ]

    return (vertices[{axis: -1}] - vertices[{axis: 0}]) / a.num_velocity


def _aia(
    a: "esis.data.Level_4",
    wavelength: u.Quantity = 304 * u.AA,
    tolerance: u.Quantity = 8 * u.s,
    center: None | na.Cartesian2dVectorArray = None,
    radius: u.Quantity = 60 * u.arcsec,
    interpolate: bool = False,
) -> tuple[np.ndarray, list[float], u.UnitBase]:
    r"""
    An AIA image of the same field at the same moment as each ESIS frame.

    AIA takes a full disk every twelve seconds and ESIS took a frame every
    ten, so there is an AIA image within a few seconds of every ESIS one, and
    the nearest is the one taken. The two instruments are not synchronized,
    so the offset wanders through the flight rather than being fixed.

    The images are registered, which turns them solar north up on a common
    plate scale of six tenths of an arcsecond and puts disk center at the
    reference pixel. That leaves a grid which is separable and identical from
    one frame to the next, so the crop to the ESIS field is worked out once
    and the rest is slicing.

    Parameters
    ----------
    a
        The Level-4 product whose frames are to be matched.
    wavelength
        The AIA channel.
    tolerance
        How far from an ESIS frame to look for an AIA one.
        Half the AIA cadence would do; a little more, so that a frame is
        found even where the two instruments drift furthest apart.
    center
        The place on the sky to crop to.
        If :obj:`None`, the whole field ESIS saw.
    radius
        The half width of the crop when `center` is given.
    interpolate
        Whether to interpolate linearly in time between the AIA frames on
        either side of each ESIS frame, rather than taking the nearest.

        The nearest frame repeats itself or skips one as the offset between
        the two cadences wanders, which is seen in a movie as a stutter.
        The frames on either side are always within one AIA cadence, so the
        window searched is widened to twice `tolerance` either side.

    Notes
    -----
    Every frame is a separate query, which is slower than asking for the
    whole flight at once but holds only one full disk in memory at a time:
    thirty of them would be four gigabytes, where thirty crops are a quarter
    of that.
    """
    axis_time = a.axis_time
    time = a.inputs.time
    num_time = time.shape[axis_time]

    unit_position = na.unit(a.inputs.position.x)
    x_min, x_max, y_min, y_max = _bounds(a, center, radius)

    images = []
    extent = None
    unit = None
    index_x = index_y = None

    for i in range(num_time):

        t = astropy.time.Time(time[{axis_time: i}].ndarray)

        window = 2 * tolerance if interpolate else tolerance

        b = sdo.aia.open(
            time_start=t - window,
            time_stop=t + window,
            wavelength=wavelength,
            register=True,
        )

        # The axis a filtergram keeps its channels on, which it names itself
        # and is whichever of its axes is not time and not a detector one.
        (axis_wavelength,) = set(b.outputs.shape) - {
            b.axis_time,
            b.axis_detector_x,
            b.axis_detector_y,
        }

        # The one AIA frame nearest this ESIS frame, of the two or three the
        # window catches.
        time_b = astropy.time.Time(np.ravel(b.inputs.time.ndarray))
        offset = (time_b - t).to_value(u.s)
        index = int(np.argmin(np.abs(offset)))

        # Or the frames either side of it and how far between them it falls,
        # when there is one either side to be had.
        weights = {index: 1.0}
        if interpolate and (offset <= 0).any() and (offset > 0).any():
            before = int(np.argmax(np.where(offset <= 0, offset, -np.inf)))
            after = int(np.argmin(np.where(offset > 0, offset, np.inf)))
            fraction = -offset[before] / (offset[after] - offset[before])
            weights = {before: 1 - fraction, after: fraction}

        index_frame = {b.axis_time: index, axis_wavelength: 0}

        if extent is None:

            # The vertices of the grid, which number one more than the cells
            # and are what a crop has to be expressed in.
            vertex = b.inputs[index_frame].position
            vertex_x = vertex.x[{b.axis_detector_y: 0}].ndarray.to(unit_position)
            vertex_y = vertex.y[{b.axis_detector_x: 0}].ndarray.to(unit_position)

            index_x = _crop(vertex_x, x_min, x_max)
            index_y = _crop(vertex_y, y_min, y_max)

            extent = [
                vertex_x[index_x.start].value,
                vertex_x[index_x.stop].value,
                vertex_y[index_y.start].value,
                vertex_y[index_y.stop].value,
            ]

        image = 0
        for index_weighted, weight in weights.items():

            outputs = b.outputs[
                {b.axis_time: index_weighted, axis_wavelength: 0}
                | {
                    b.axis_detector_x: index_x,
                    b.axis_detector_y: index_y,
                }
            ]

            unit = na.unit(outputs)

            # Named rather than assumed, since `imshow` reads the first axis
            # as the one running up the screen. Cast down on the way out: a
            # count is not worth eight bytes, and thirty frames of them would
            # be worth two gigabytes.
            outputs = outputs.transpose(
                axes=(b.axis_detector_y, b.axis_detector_x),
            )
            frame = outputs.ndarray.value.astype(np.float32)
            image = frame if weight == 1 else image + np.float32(weight) * frame

        images.append(image)

    return np.stack(images), extent, unit


def _hmi(
    a: "esis.data.Level_4",
    center: None | na.Cartesian2dVectorArray = None,
    radius: u.Quantity = 60 * u.arcsec,
    series: str = "hmi.M_45s",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, u.UnitBase]:
    r"""
    An HMI magnetogram of the same field at the same moment as each ESIS
    frame.

    HMI takes forty five seconds to build a magnetogram where ESIS took ten
    to take a frame, so there is no image of its own for each: about seven
    magnetograms cover the whole flight, and each stands for the four or five
    ESIS frames nearest it. The field is what it is doing over the flight
    rather than what it is doing in a given second, which is the right way
    round for it, since the field of the quiet Sun does not turn over in
    five minutes.

    Unlike AIA there is no registration step, so the grid is what HMI
    recorded on: a rotation of a hundred and eighty degrees and a fraction,
    which makes the coordinates of a pixel depend on both of its indices. The
    positions are therefore returned as a mesh rather than as an extent.

    Parameters
    ----------
    a
        The Level-4 product whose frames are to be matched.
    center
        The place on the sky to crop to.
        If :obj:`None`, the whole field ESIS saw.
    radius
        The half width of the crop when `center` is given.
    series
        The HMI series to take the magnetograms from.

    Returns
    -------
    The magnetograms, one per ESIS frame, the :math:`x` and :math:`y` of the
    vertices of the grid they are on, and their unit.
    """
    time = astropy.time.Time(np.ravel(a.inputs.time.ndarray))

    x_min, x_max, y_min, y_max = _bounds(a, center, radius)

    # One query for the whole flight, since a magnetogram is shared between
    # frames and asking per frame would be asking for the same one again.
    b = sdo.hmi.open(
        time_start=time.min(),
        time_stop=time.max(),
        series=series,
    )

    axis_x = b.axis_detector_x
    axis_y = b.axis_detector_y

    time_b = astropy.time.Time(np.ravel(b.inputs.time.ndarray))

    # The pointing holds still over five minutes, so where the region falls
    # on the detector is worked out once, from the first magnetogram.
    vertex = b.inputs[{b.axis_time: 0}].position
    vertex_x = vertex.x.to(na.unit(a.inputs.position.x))
    vertex_y = vertex.y.to(na.unit(a.inputs.position.y))

    inside = (
        (vertex_x > x_min)
        & (vertex_x < x_max)
        & (vertex_y > y_min)
        & (vertex_y < y_max)
    )

    index = {
        axis_x: _extent_true(inside.any(axis=axis_y)),
        axis_y: _extent_true(inside.any(axis=axis_x)),
    }

    # One more vertex than there are cells, on each axis.
    index_vertex = {ax: slice(index[ax].start, index[ax].stop + 1) for ax in index}

    x = vertex_x[index_vertex].transpose(axes=(axis_y, axis_x)).ndarray.value
    y = vertex_y[index_vertex].transpose(axes=(axis_y, axis_x)).ndarray.value

    unit = na.unit(b.outputs)

    images = []
    for i in range(time.size):
        j = int(np.argmin(np.abs((time_b - time[i]).to_value(u.s))))
        outputs = b.outputs[{b.axis_time: j} | index]
        outputs = outputs.transpose(axes=(axis_y, axis_x))
        images.append(outputs.ndarray.value.astype(np.float32))

    return np.stack(images), x, y, unit


def _extent_true(condition: na.AbstractScalarArray) -> slice:
    """
    The run of indices from the first true to the last.

    Parameters
    ----------
    condition
        A one-dimensional boolean array.
    """
    (where,) = np.nonzero(condition.ndarray)
    return slice(int(where[0]), int(where[-1]) + 1)


def _megameters_per_arcsec(time: astropy.time.Time) -> float:
    r"""
    How far an arcsecond is on the Sun, on a given day, in megameters.

    The Earth's distance from the Sun changes by three percent over a year,
    so this is not a constant and is worked out for the day rather than
    assumed. What it gives is a distance across the sky: a length on the
    surface is longer than this by the foreshortening, which is half a
    percent ten degrees from the middle of the disk and can be ignored there.

    Parameters
    ----------
    time
        The day to work it out for.
    """
    distance = astropy.coordinates.get_sun(time).distance
    return (distance * (1 * u.arcsec).to_value(u.rad)).to_value(u.Mm)


def _scale_bar(
    ax: "matplotlib.axes.Axes",
    length: float,
    label: str,
    margin: float = 0.06,
) -> None:
    """
    A bar of a given length in the upper left of a panel, with its label.

    White and outlined in black, so that it can be read against whatever
    the panel shows there, dark or light.

    Parameters
    ----------
    ax
        The panel to draw it on, whose limits have already been set.
    length
        The length of the bar, in the units of the horizontal axis.
    label
        What to write under the bar.
    margin
        How far in from the corner to put the left end of the bar, as a
        fraction of the width and of the height of the panel.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + margin * (x1 - x0)
    y = y1 - margin * (y1 - y0)
    outline = [matplotlib.patheffects.withStroke(linewidth=4, foreground="black")]
    ax.plot(
        [x, x + length],
        [y, y],
        color="white",
        linewidth=2.5,
        solid_capstyle="butt",
        path_effects=outline,
        scalex=False,
        scaley=False,
    )
    ax.text(
        x + length / 2,
        y - 0.02 * (y1 - y0),
        label,
        color="white",
        ha="center",
        va="top",
        fontsize="small",
        path_effects=[
            matplotlib.patheffects.withStroke(linewidth=3, foreground="black"),
        ],
    )


def _crop(vertex: u.Quantity, minimum: u.Quantity, maximum: u.Quantity) -> slice:
    """
    The cells of a grid which cover a span.

    Parameters
    ----------
    vertex
        The vertices of the grid to crop, in increasing order.
    minimum
        The low end of the span to cover.
    maximum
        The high end of the span to cover.
    """
    start = int(np.searchsorted(vertex, minimum, side="right")) - 1
    stop = int(np.searchsorted(vertex, maximum, side="left"))
    start = max(start, 0)
    stop = min(stop, vertex.size - 1)
    return slice(start, stop)


def _bounds(
    a: "esis.data.Level_4",
    center: None | na.Cartesian2dVectorArray,
    radius: u.Quantity,
) -> tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity]:
    """
    The corners of the region to show, as an instrument-independent box.

    Parameters
    ----------
    a
        The Level-4 product, whose field is the whole region when no center
        is given.
    center
        The place on the sky to look at, or :obj:`None` for the whole field.
    radius
        The half width of the region when `center` is given.
    """
    position = a.inputs.position
    if center is None:
        return (
            position.x.min().ndarray,
            position.x.max().ndarray,
            position.y.min().ndarray,
            position.y.max().ndarray,
        )
    return (
        center.x - radius,
        center.x + radius,
        center.y - radius,
        center.y + radius,
    )


def _crop_esis(
    a: "esis.data.Level_4",
    center: None | na.Cartesian2dVectorArray,
    radius: u.Quantity,
) -> tuple[dict[str, slice], list[float]]:
    """
    The cells of the ESIS field covering a region, and the region they cover.

    The grid is separable and uniform, so the crop is a pair of slices and
    what it covers is an extent an image can be drawn in.

    Parameters
    ----------
    a
        The Level-4 product to crop.
    center
        The place on the sky to look at, or :obj:`None` for the whole field.
    radius
        The half width of the region when `center` is given.
    """
    x_min, x_max, y_min, y_max = _bounds(a, center, radius)

    vertex_x = a.inputs.position.x.ndarray
    vertex_y = a.inputs.position.y.ndarray

    index_x = _crop(vertex_x, x_min, x_max)
    index_y = _crop(vertex_y, y_min, y_max)

    extent = [
        vertex_x[index_x.start].value,
        vertex_x[index_x.stop].value,
        vertex_y[index_y.start].value,
        vertex_y[index_y.stop].value,
    ]

    return {a.axis_x: index_x, a.axis_y: index_y}, extent


def _energy_photon(a: "esis.data.Level_4", index_line: int) -> u.Quantity:
    """
    What one photon of a line is worth.

    The product counts photons, and a photon of one line is not worth the same
    energy as a photon of another, so asking for an energy is asking for a
    conversion which differs from line to line: eight percent separates the
    ends of this range. It is done at the line's rest wavelength, which is
    good to a part in a thousand across a window.

    Parameters
    ----------
    a
        The Level-4 product the line belongs to.
    index_line
        The index of the spectral line.
    """
    wavelength = a.wavelength_center.ndarray[index_line]
    h = astropy.constants.h
    c = astropy.constants.c
    return (h * c / wavelength).to(u.erg) / u.photon


def level_4_lines(
    path_data: None | pathlib.Path = None,
    wavelength_aia: None | u.Quantity = 304 * u.AA,
    unit_intensity: u.UnitBase = u.erg / (u.s * u.cm**2 * u.deg**2),
    percentile: float = 99.5,
    cmap: str = "gray",
    ncols: int = 3,
    figsize: tuple[float, float] = figsize_default,
    dpi: float = 150,
    fps: int = 5,
    timestamp: bool = True,
    suffix: str = ".mp4",
    path: None | pathlib.Path = None,
) -> pathlib.Path:
    """
    Loop over the intensity of every line ESIS recovered.

    The Level-4 product is inverted one spectral window at a time and
    distributed one file per window, and a whole directory of them reads back
    as a single product of several lines. This shows the intensity of each,
    the light of that window integrated over its Doppler range, so that the
    same moment of the flight can be seen in every line at once.

    The lines are formed at very different temperatures, from the
    chromosphere to the corona, so what changes between the panels is which
    part of the atmosphere is being looked at rather than which part of the
    Sun.

    Each panel is scaled to itself. The lines differ in brightness by far
    more than the structure within any one of them, so a single scale would
    leave the faint windows black; the price is that the panels say which
    structures appear in which line rather than which line is brighter.

    The last panel is AIA rather than ESIS, cropped to the same field and
    matched frame for frame. It is there to be recognized: AIA 304 is a
    channel everybody has looked at, so it says whether what ESIS recovered
    is the Sun that was there, and its own line is He II 304, which is formed
    near where He I 584 is and can be read against it.

    Parameters
    ----------
    path_data
        The directory of Level-4 files to read.
        If :obj:`None`, the directory the local copies are kept in.
    wavelength_aia
        The AIA channel drawn in the panel after the lines.
        If :obj:`None`, that panel is left empty and no AIA data is fetched.
    unit_intensity
        The unit to draw the intensities in, see :func:`_energy_photon`.
    percentile
        The percentile of each line placed at the top of its own brightness
        scale.
    cmap
        The colormap.
    ncols
        The number of panels across.
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution of the saved animation in dots per inch.
    fps
        The number of frames per second in the saved animation.
    timestamp
        Whether to write the time of each exposure above the panels.
    suffix
        The file type of the animation, either ``".mp4"`` or ``".gif"``.
        Ignored if `path` is given.
    path
        The location to save the animation.
        If :obj:`None`, it is saved alongside the other figures.
    """
    if path_data is None:
        path_data = path_level_4_default.parent

    a = esis.data.Level_4.from_fits(path_data)

    axis_line = a.axis_line
    axis_time = a.axis_time

    intensity = _intensity(a)
    num_line = intensity.shape[axis_line]
    num_time = intensity.shape[axis_time]

    position = a.inputs.position
    unit_position = na.unit(position.x)

    # A uniform grid, so the panels can be drawn as images, which is both
    # quicker than a mesh and cheap to update: only the array behind each
    # image changes from one frame to the next.
    extent = [
        position.x.min().ndarray.value,
        position.x.max().ndarray.value,
        position.y.min().ndarray.value,
        position.y.max().ndarray.value,
    ]

    # Coolest first, so that reading across the panels is reading up through
    # the atmosphere rather than along the spectrum.
    order = order_temperature(list(a.label_line))

    # Frame first and then the axis running up the screen, which is the order
    # `imshow` reads, so that drawing a frame is a plain index.
    images = [
        (
            intensity[{axis_line: i}]
            .transpose(axes=(axis_time, a.axis_y, a.axis_x))
            .ndarray
            * _energy_photon(a, i)
        ).to_value(unit_intensity)
        for i in order
    ]
    extents = [extent] * num_line
    labels = [a.label_line[i] for i in order]
    units = [unit_intensity] * num_line

    # AIA is another instrument on another grid, so it is fetched and cropped
    # rather than sliced out of the product, but from here on it is one more
    # panel like the rest.
    if wavelength_aia is not None:
        images_aia, extent_aia, unit_aia = _aia(a, wavelength_aia)
        images.append(images_aia)
        extents.append(extent_aia)
        labels.append(f"AIA {wavelength_aia.to_value(u.AA):.0f}")
        units.append(unit_aia)

    num_panel = len(images)

    # Each panel against its own brightest places, over the whole flight, so
    # that it neither fades with the signal nor is set by one frame.
    vmax = [float(np.nanpercentile(im, percentile)) for im in images]

    time = a.inputs.time

    nrows = -(-num_panel // ncols)

    if path is None:
        path = default_path / f"level-4-lines{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axs = np.asarray(axs).ravel()

    artists = []
    for i in range(num_panel):
        ax = axs[i]
        artists.append(
            ax.imshow(
                images[i][0],
                origin="lower",
                extent=extents[i],
                cmap=cmap,
                vmin=0,
                vmax=vmax[i],
                aspect="equal",
            )
        )
        ax.set_title(labels[i])
        fig.colorbar(
            artists[i],
            ax=ax,
            fraction=0.045,
            pad=0.02,
            label=f"intensity ({units[i]:latex_inline})",
        )

    # The panels nothing landed in have nothing to say.
    for ax in axs[num_panel:]:
        ax.set_axis_off()

    # Every panel is cropped to the field ESIS saw, including the AIA one,
    # which arrives as a piece of a full disk and would otherwise set the
    # limits of all of them: the axes are shared.
    axs[0].set_xlim(extent[0], extent[1])
    axs[0].set_ylim(extent[2], extent[3])

    # The lowest panel of each column, which is not the bottom row where a
    # column runs out early: `sharex` hides the tick labels of every panel
    # with another below it, so a panel with an empty one below would
    # otherwise be left with no scale at all.
    for column in range(ncols):
        index = max(i for i in range(num_panel) if i % ncols == column)
        axs[index].set_xlabel(
            f"helioprojective $x$ ({unit_position:latex_inline})",
        )
        axs[index].tick_params(labelbottom=True)
    for ax in axs[::ncols]:
        ax.set_ylabel(f"helioprojective $y$ ({unit_position:latex_inline})")

    # A suptitle rather than free text at the top of the figure, so that the
    # layout leaves room for it instead of laying it over the titles.
    if timestamp:
        text = fig.suptitle("")
    else:
        text = None

    def func(index: int) -> list[matplotlib.artist.Artist]:
        for i, image in enumerate(artists):
            image.set_data(images[i][index])
        if text is None:
            return artists
        t = time[{axis_time: index}].ndarray
        text.set_text(t.strftime("%Y-%m-%d %H:%M:%S UTC"))
        return [*artists, text]

    if path.suffix == ".gif":
        writer = matplotlib.animation.PillowWriter(fps=fps)
    else:
        writer = matplotlib.animation.FFMpegWriter(
            fps=fps,
            codec="h264",
            extra_args=["-pix_fmt", "yuv420p", "-crf", "18"],
        )

    ani = matplotlib.animation.FuncAnimation(fig=fig, func=func, frames=num_time)
    ani.save(filename=path, writer=writer, dpi=dpi)

    plt.close(fig)

    return path


def level_4_event(
    path_data: None | pathlib.Path = None,
    center: None | na.Cartesian2dVectorArray = None,
    radius: u.Quantity = radius_event_default,
    wavelength_aia: u.Quantity = [304, 131, 171, 193] * u.AA,
    interpolate_aia: bool = False,
    context_top: bool = False,
    unit_intensity: u.UnitBase = u.erg / (u.s * u.cm**2 * u.deg**2),
    percentile: float = 99.5,
    gamma: float = 1,
    velocity_limit: u.Quantity = 60 * u.km / u.s,
    magnetogram_limit: u.Quantity = 100 * u.G,
    cmap: str = "gray",
    cmap_velocity: str = "RdBu_r",
    color_bad: str = "black",
    figsize: tuple[float, float] = figsize_default,
    dpi: float = 150,
    fps: int = 5,
    timestamp: bool = True,
    suffix: str = ".mp4",
    path: None | pathlib.Path = None,
) -> pathlib.Path:
    r"""
    Loop over one event in every line ESIS recovered, and in context.

    The whole of what was measured of one place on the Sun, at once. The top
    row is the intensity of each line, the middle row is the Doppler shift of
    the same line directly beneath it, and the bottom row is what the two
    instruments watching the same place from orbit saw: AIA in its
    :math:`304\,\text{\AA}` channel, two coronal ones, and an HMI magnetogram.

    Reading down a column says what one line is doing and how fast; reading
    across the top row says which temperatures it is doing it at; and the
    bottom row says what the field there looks like, which is what an
    explosive event is supposed to be about.

    The rows are scaled differently on purpose. Each intensity panel is
    scaled to itself, since the lines differ in brightness by far more than
    the structure within any one of them. Every Doppler panel shares one
    scale, so a color means one velocity across the whole row and the lines
    can be read against each other.

    Parameters
    ----------
    path_data
        The directory of Level-4 files to read.
        If :obj:`None`, the directory the local copies are kept in.
    center
        The place on the sky to look at.
        If :obj:`None`, :obj:`position_event_e`.
    radius
        The half width of the region shown.
    wavelength_aia
        The AIA channels drawn in the bottom row, one panel each.

        Four by default, in order of the temperature they are formed at,
        as the rows above them are: 304 is the He II line, formed near the
        coolest of the ESIS lines, 131 sits between that and the rest, and
        171 and 193 are coronal, formed above the hottest of them.
    interpolate_aia
        Whether to interpolate the AIA images linearly in time onto each
        ESIS frame, rather than taking the nearest, which stutters as the
        twelve-second AIA cadence drifts against the ten-second ESIS one.
        See :func:`_aia`.
    context_top
        Whether to put the row of AIA and HMI at the top rather than the
        bottom, so that the context is seen first and the ESIS rows are read
        against it.
    unit_intensity
        The unit to draw the intensities in, see :func:`_energy_photon`.
    percentile
        The percentile of each line placed at the top of its own brightness
        scale.
    gamma
        The power the intensities of the top row and of AIA are raised to
        before they are turned into a brightness, as in
        :func:`level_4_event_history`. One, the default, is linear. One half
        is the square root, which keeps the network around an event visible
        when the top of the scale is high enough not to clip the event.
    velocity_limit
        The Doppler velocity at each end of the middle row's colormap.
    magnetogram_limit
        The field strength at each end of the magnetogram's colormap.
        Far below the strongest fields present, since otherwise everything
        but the strongest is the same shade of grey.
    cmap
        The colormap of the intensities and of the magnetogram.
    cmap_velocity
        The colormap of the Doppler shifts.
        One which runs through white at its middle, since the quantity is
        signed and its middle is where the plasma is at rest.
    color_bad
        The color of the places with no Doppler shift to report.
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution of the saved animation in dots per inch.
    fps
        The number of frames per second in the saved animation.
    timestamp
        Whether to write the time of each exposure above the panels.
    suffix
        The file type of the animation, either ``".mp4"`` or ``".gif"``.
        Ignored if `path` is given.
    path
        The location to save the animation.
        If :obj:`None`, it is saved alongside the other figures.

    Notes
    -----
    The three rows are not on the same clock. ESIS took a frame every ten
    seconds and AIA takes one every twelve, so those two step together, but a
    magnetogram takes forty five seconds to build and one of them therefore
    stands for several ESIS frames at a time.
    """
    if path_data is None:
        path_data = path_level_4_default.parent

    if center is None:
        center = center_event_default

    a = esis.data.Level_4.from_fits(path_data)

    axis_time = a.axis_time
    axis_wavelength = a.axis_wavelength

    num_line = a.num_line
    num_time = a.outputs.shape[axis_time]

    index, extent = _crop_esis(a, center, radius)

    unit_position = na.unit(a.inputs.position.x)
    unit_velocity = velocity_limit.unit

    # The edges of the velocity bins, not their centers: `na.pdf.median`
    # reads `x` as the edges and wants one more of them than there are
    # values, and passing the centers instead shifts every median by half a
    # cell.
    velocity = a.velocity

    # Worked out for the whole flight before anything is drawn, which is
    # cheap once the region is only a couple of arcminutes across, and saves
    # recomputing a frame each time the animation comes back to it.
    order = order_temperature(list(a.label_line))

    intensity = []
    shift = []
    for i in order:

        radiance = a.outputs[a.window(i) | index]

        moment_0 = radiance.sum(axis_wavelength) * _width(a, i)
        intensity.append(
            (
                moment_0.transpose(axes=(axis_time, a.axis_y, a.axis_x)).ndarray
                * _energy_photon(a, i)
            ).to_value(unit_intensity)
        )

        median = na.pdf.median(
            x=velocity,
            f=radiance,
            axis=axis_wavelength,
        )
        shift.append(
            median.transpose(axes=(axis_time, a.axis_y, a.axis_x))
            .to_value(unit_velocity)
            .ndarray
        )

    vmax = [float(np.nanpercentile(im, percentile)) for im in intensity]

    images_aia = []
    extent_aia = []
    vmax_aia = []
    for wavelength_aia_i in wavelength_aia:
        images_i, extent_i, _ = _aia(
            a=a,
            wavelength=wavelength_aia_i,
            center=center,
            radius=radius,
            interpolate=interpolate_aia,
        )
        images_aia.append(images_i)
        extent_aia.append(extent_i)
        vmax_aia.append(float(np.nanpercentile(images_i, percentile)))

    num_aia = len(images_aia)

    images_hmi, x_hmi, y_hmi, unit_hmi = _hmi(
        a=a,
        center=center,
        radius=radius,
    )

    time = a.inputs.time

    colormap_velocity = plt.get_cmap(cmap_velocity).with_extremes(bad=color_bad)
    norm_velocity = matplotlib.colors.Normalize(
        vmin=-velocity_limit.to_value(unit_velocity),
        vmax=+velocity_limit.to_value(unit_velocity),
    )
    norm_magnetogram = matplotlib.colors.Normalize(
        vmin=-magnetogram_limit.to_value(unit_hmi),
        vmax=+magnetogram_limit.to_value(unit_hmi),
    )

    if path is None:
        path = default_path / f"level-4-event{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(
        nrows=3,
        ncols=num_line,
        figsize=figsize,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    # Room down the right for the color scale of the velocity row, which has
    # no empty panel of its own to go in.
    fig.get_layout_engine().set(rect=(0, 0, 0.915, 1))

    # Which row of the figure each kind of panel is in.
    if context_top:
        row_context, row_intensity, row_velocity = 0, 1, 2
    else:
        row_intensity, row_velocity, row_context = 0, 1, 2

    images_line = []

    def _scale(vmax_i: float) -> dict:
        """The brightness scale of an intensity panel, linear unless asked."""
        if gamma == 1:
            return dict(vmin=0, vmax=vmax_i)
        return dict(norm=matplotlib.colors.PowerNorm(gamma=gamma, vmin=0, vmax=vmax_i))

    for i in range(num_line):

        images_line.append(
            axs[row_intensity, i].imshow(
                intensity[i][0],
                origin="lower",
                extent=extent,
                cmap=cmap,
                **_scale(vmax[i]),
                aspect="equal",
            )
        )
        axs[row_intensity, i].set_title(a.label_line[order[i]])

        images_line.append(
            axs[row_velocity, i].imshow(
                shift[i][0],
                origin="lower",
                extent=extent,
                cmap=colormap_velocity,
                norm=norm_velocity,
                aspect="equal",
            )
        )

    image_aia = []
    for i in range(num_aia):
        image_aia.append(
            axs[row_context, i].imshow(
                images_aia[i][0],
                origin="lower",
                extent=extent_aia[i],
                cmap=cmap,
                **_scale(vmax_aia[i]),
                aspect="equal",
            )
        )
        axs[row_context, i].set_title(f"AIA {wavelength_aia[i].to_value(u.AA):.0f}")

    # A mesh rather than an image, since HMI is not registered and the
    # coordinates of a pixel therefore depend on both of its indices.
    ax_hmi = axs[row_context, num_aia]
    mesh_hmi = ax_hmi.pcolormesh(
        x_hmi,
        y_hmi,
        images_hmi[0],
        cmap=cmap,
        norm=norm_magnetogram,
        rasterized=True,
    )
    ax_hmi.set_title("HMI magnetogram")
    ax_hmi.set_aspect("equal")

    # The panels of the bottom row which no instrument landed in.
    for ax in axs[row_context, num_aia + 1 :]:
        ax.set_axis_off()

    # Every panel shows the same piece of sky and the axes are shared, so
    # this is said once. The AIA and HMI grids reach a little past the ESIS
    # one, and are clipped to it rather than being allowed to widen it.
    axs[0, 0].set_xlim(extent[0], extent[1])
    axs[0, 0].set_ylim(extent[2], extent[3])

    label_row = (
        f"intensity ({unit_intensity:latex_inline})",
        f"LOS velocity ({unit_velocity:latex_inline})",
        "context",
    )
    for row, label in zip((row_intensity, row_velocity, row_context), label_row):
        axs[row, 0].set_ylabel(label)

    # The lowest panel of each column, which is the middle row for the
    # columns the bottom row does not reach: `sharex` hides the tick labels
    # of every panel with another below it, and a panel with an empty one
    # below would otherwise be left with no scale at all.
    for column in range(num_line):
        if row_context == 2 and column <= num_aia:
            row = row_context
        else:
            row = row_velocity
        axs[row, column].set_xlabel(
            f"helioprojective $x$ ({unit_position:latex_inline})",
        )
        axs[row, column].tick_params(labelbottom=True)

    if timestamp:
        text = fig.suptitle(
            time[{axis_time: 0}].ndarray.strftime("%Y-%m-%d %H:%M:%S UTC")
        )
    else:
        text = None

    # Placed against the panels rather than in the layout, which is why the
    # positions are asked for only once everything else has settled: a
    # vertical scale beside the row it describes, and the magnetogram's
    # beside the magnetogram.
    fig.canvas.draw()

    position = axs[row_velocity, -1].get_position()
    fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm_velocity, cmap=colormap_velocity),
        cax=fig.add_axes((position.x1 + 0.010, position.y0, 0.011, position.height)),
        label=f"LOS velocity ({unit_velocity:latex_inline})",
    )

    position = ax_hmi.get_position()
    fig.colorbar(
        mesh_hmi,
        cax=fig.add_axes((position.x1 + 0.010, position.y0, 0.011, position.height)),
        label=f"line-of-sight field ({unit_hmi:latex_inline})",
    )

    # Held still now, so that the panels do not shift about from one frame to
    # the next and leave the scales beside nothing.
    fig.set_layout_engine("none")

    def func(index_time: int) -> list[matplotlib.artist.Artist]:

        for i in range(num_line):
            images_line[2 * i].set_data(intensity[i][index_time])
            images_line[2 * i + 1].set_data(shift[i][index_time])

        for i in range(num_aia):
            image_aia[i].set_data(images_aia[i][index_time])
        mesh_hmi.set_array(images_hmi[index_time])

        result = [*images_line, *image_aia, mesh_hmi]

        if text is None:
            return result

        t = time[{axis_time: index_time}].ndarray
        text.set_text(t.strftime("%Y-%m-%d %H:%M:%S UTC"))

        return [*result, text]

    if path.suffix == ".gif":
        writer = matplotlib.animation.PillowWriter(fps=fps)
    else:
        writer = matplotlib.animation.FFMpegWriter(
            fps=fps,
            codec="h264",
            extra_args=["-pix_fmt", "yuv420p", "-crf", "18"],
        )

    ani = matplotlib.animation.FuncAnimation(fig=fig, func=func, frames=num_time)
    ani.save(filename=path, writer=writer, dpi=dpi)

    plt.close(fig)

    return path


def _order_gaussians(double: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    The two Gaussians of a fit, put in the order core and then second.

    The core is the one nearer to rest, since the core of the line is the
    plasma which is not taking part in the event and the event is what
    moves. That only means something when the two are apart, though: two
    Gaussians overlapping near rest are one line split in two, and which of
    them is nearer to rest changes from one place to the next with the
    noise. Where the centers are closer than the sum of the widths, the core
    is instead the brighter one.

    Parameters
    ----------
    double
        The pair of Gaussians from :func:`spd2026._gaussians.fit_gaussians`,
        with their eight numbers along the last axis.

    Returns
    -------
    core
        The area, center and width of the core, along the last axis.
    second
        The same of the other one.
    """
    first, other = double[..., 0:3], double[..., 3:6]
    area_1, center_1, width_1 = np.moveaxis(first, ~0, 0)
    area_2, center_2, width_2 = np.moveaxis(other, ~0, 0)

    apart = np.abs(center_1 - center_2) > width_1 + width_2
    first_is_core = np.where(
        apart,
        np.abs(center_1) <= np.abs(center_2),
        area_1 >= area_2,
    )[..., np.newaxis]

    return (
        np.where(first_is_core, first, other),
        np.where(first_is_core, other, first),
    )


def _gaussians(
    a: "esis.data.Level_4",
    label_line: str,
    center: na.Cartesian2dVectorArray,
    radius: u.Quantity,
    width_min: u.Quantity,
    width_max: u.Quantity,
    num_passes: int,
    unit_velocity: u.UnitBase = u.km / u.s,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Two Gaussians fitted to every profile of one line in a region, core first.

    Parameters
    ----------
    a
        The Level-4 product to fit.
    label_line
        The line to fit.
    center
        The middle of the region to fit.
    radius
        The half width of the region to fit.
    width_min
        The narrowest a Gaussian may be, as a standard deviation.
    width_max
        The broadest a Gaussian may be, as a standard deviation.
    num_passes
        The most passes to make fitting each pair again from the pairs of
        its neighbors, see :func:`spd2026._gaussians.seed_gaussians`. Zero
        leaves each fit as it was made on its own.
    unit_velocity
        The unit the centers, the widths and the edges are returned in.

    Returns
    -------
    core
        The area, center and width of the core, see :func:`_order_gaussians`,
        with time, :math:`y` and :math:`x` as the first three axes, as the
        whole flight.
    second
        The same of the second component.
    edges
        The edges of the cells of the spectrum the fits were made on.
    """
    axis_wavelength = a.axis_wavelength

    index_line = list(a.label_line).index(label_line)
    index, _ = _crop_esis(a, center, radius)

    # Contiguous, since the cache recognizes an array by its layout in
    # memory as well as by its values, and a transposed view would be a new
    # array to it every time.
    radiance = np.ascontiguousarray(
        a.outputs[a.window(index_line) | index]
        .transpose(axes=(a.axis_time, a.axis_y, a.axis_x, axis_wavelength))
        .ndarray.value
    )

    # The edges of the cells, one set shared by every line, see
    # :func:`level_4_event_history`.
    velocity_edges = a.velocity[
        {k: 0 for k in a.velocity.shape if k != axis_wavelength}
    ]
    edges = np.asarray(velocity_edges.ndarray.to_value(unit_velocity))

    kwargs = dict(
        radiance=radiance,
        edges=edges,
        width_min=float(width_min.to_value(unit_velocity)),
        width_max=float(width_max.to_value(unit_velocity)),
    )
    if num_passes:
        _, double = seed_gaussians(**kwargs, num_passes=num_passes)
    else:
        _, double = fit_gaussians(**kwargs)

    core, second = _order_gaussians(double)

    return core, second, edges


width_thermal_o_v = np.sqrt(
    astropy.constants.k_B * temperature_line["O V 630"] / (15.999 * u.u)
).to(u.km / u.s)
"""
The narrowest a Gaussian fitted to O V is allowed to be: the thermal width of
the line, the spread of the speeds of oxygen ions along the line of sight at
the temperature the line is formed at, as a standard deviation.

Nothing emitting O V can be narrower, and without a floor the second Gaussian
of a fit is free to shrink onto a single cell of the spectrum and fit the
noise of the inversion there instead of any plasma: a quarter of them did, in
a first try with a floor of half a cell.
"""


def level_4_event_gaussians(
    path_data: None | pathlib.Path = None,
    center: None | na.Cartesian2dVectorArray = None,
    radius: u.Quantity = radius_event_default,
    num_frames_dropped: int = 1,
    label_line: str = "O V 630",
    width_min: u.Quantity = width_thermal_o_v,
    width_max: u.Quantity = 150 * u.km / u.s,
    num_passes: int = 10,
    unit_intensity: u.UnitBase = u.erg / (u.s * u.cm**2 * u.deg**2),
    percentile: float = 99.99,
    gamma: float = 0.5,
    velocity_limit_core: u.Quantity = 60 * u.km / u.s,
    velocity_limit_second: u.Quantity = 150 * u.km / u.s,
    width_limit: u.Quantity = 80 * u.km / u.s,
    scale_bar: None | u.Quantity = 10 * u.Mm,
    cmap: str = "gray",
    cmap_velocity: str = "RdBu_r",
    cmap_width: str = "viridis",
    figsize: tuple[float, float] = figsize_default,
    dpi: float = 150,
    fps: int = 5,
    timestamp: bool = True,
    suffix: str = ".mp4",
    path: None | pathlib.Path = None,
) -> pathlib.Path:
    r"""
    Two Gaussians fitted to every profile of one line around an event, as a
    loop over the flight.

    Each profile is fitted with a pair of Gaussians on a flat background, see
    :func:`spd2026._gaussians.fit_gaussians`, and the six numbers the pair
    comes to are drawn as six maps: the intensity, Doppler shift and width
    of the core of the line in the top row, and the same of the second
    Gaussian beneath them. The core is the Gaussian nearer to rest, which is
    what the fast flows of an explosive event stand out from, see
    :func:`_order_gaussians`.

    The intensities share one brightness scale, so the two rows can be read
    against each other and the second Gaussian's share of the line seen
    directly. The Doppler shifts do not share a scale: the core moves by
    tens of kilometers per second and the second Gaussian by more than a
    hundred, and on one scale the core would be white throughout.

    Parameters
    ----------
    path_data
        The directory of Level-4 files to read.
        If :obj:`None`, the directory the local copies are kept in.
    center
        The place on the sky to look at.
        If :obj:`None`, :obj:`center_event_default`, the region the other
        close-ups of event E show.
    radius
        The half width of the region shown.
    num_frames_dropped
        How many frames to leave off the end of the flight, see
        :func:`level_4_event_history`. In the last frame the fits find
        components everywhere, at every speed the window holds, in a line
        which is only a tenth as bright as it was at the top of the flight.
    label_line
        The line to fit.
    width_min
        The narrowest a Gaussian may be, as a standard deviation. The
        default is right for O V only, see :obj:`width_thermal_o_v`.
    width_max
        The broadest a Gaussian may be, as a standard deviation.
    num_passes
        The most passes to make fitting each pair again from the pairs of
        its neighbors, see :func:`spd2026._gaussians.seed_gaussians`, which
        is what keeps the maps from being speckled with places the fit went
        a different way than the places around them. Zero leaves each fit
        as it was made on its own.
    unit_intensity
        The unit to draw the intensities in, see :func:`_energy_photon`.
    percentile
        The percentile of the total intensity of the two Gaussians placed at
        the top of the brightness scale.
    gamma
        The power the intensities are raised to before they are turned into
        a brightness, as in :func:`level_4_event_history`.
    velocity_limit_core
        The Doppler velocity at each end of the colormap of the core.
    velocity_limit_second
        The Doppler velocity at each end of the colormap of the second
        Gaussian.
    width_limit
        The width at the top of the colormap of both widths, which share it
        so that a color is one width in either row. The second component of
        a fast flow is typically thirty to seventy kilometers per second
        wide, and the core twenty to thirty.
    scale_bar
        The length of the bar drawn in the upper left of every panel, so
        that sizes can be read off in megameters, or :obj:`None` for none.

        A bar rather than a second set of axes in megameters, which the
        panels of :func:`level_4_event_history` have, since here there are
        six panels, each with a color scale of its own down its right side,
        and nowhere to put them.
    cmap
        The colormap of the intensities.
    cmap_velocity
        The colormap of the Doppler shifts.
    cmap_width
        The colormap of the widths.
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution of the saved animation in dots per inch.
    fps
        The number of frames per second in the saved animation.
    timestamp
        Whether to write the time of each exposure above the panels.
    suffix
        The file type of the animation, either ``".mp4"`` or ``".gif"``.
        Ignored if `path` is given.
    path
        The location to save the animation.
        If :obj:`None`, it is saved alongside the other figures.

    Notes
    -----
    The widths are what the fit found, with nothing taken out for the
    instrument or the inversion: they are upper limits on the width of the
    plasma, not measurements of it.

    The spectral window of each line stops at :math:`\pm 210` km/s, and a
    Gaussian centered far out in a wing has part of itself beyond the edge.
    The fit accounts for the part it cannot see, since it compares only the
    cells there are, but it has less to go on there, and a second Gaussian
    past about 150 km/s is poorly constrained.
    """
    if path_data is None:
        path_data = path_level_4_default.parent

    if center is None:
        center = center_event_default

    a = esis.data.Level_4.from_fits(path_data)

    axis_time = a.axis_time

    index_line = list(a.label_line).index(label_line)
    _, extent = _crop_esis(a, center, radius)

    unit_position = na.unit(a.inputs.position.x)
    unit_velocity = velocity_limit_core.unit

    core, second, _ = _gaussians(
        a=a,
        label_line=label_line,
        center=center,
        radius=radius,
        width_min=width_min,
        width_max=width_max,
        num_passes=num_passes,
        unit_velocity=unit_velocity,
    )

    # Fitted, so that the cache holds the whole flight, but not shown.
    num_time = core.shape[0] - num_frames_dropped
    core = core[:num_time]
    second = second[:num_time]

    # The areas are sums of cells, which is what the product's intensity is
    # before the width of a cell and the energy of a photon are put in.
    factor = (
        na.unit(a.outputs)
        * _width(a, index_line).ndarray
        * _energy_photon(a, index_line)
    ).to_value(unit_intensity)

    components = [
        dict(
            name="core",
            intensity=core[..., 0] * factor,
            velocity=core[..., 1],
            width=core[..., 2],
            velocity_limit=velocity_limit_core.to_value(unit_velocity),
        ),
        dict(
            name="second component",
            intensity=second[..., 0] * factor,
            velocity=second[..., 1],
            width=second[..., 2],
            velocity_limit=velocity_limit_second.to_value(unit_velocity),
        ),
    ]

    vmax = float(
        np.nanpercentile(
            components[0]["intensity"] + components[1]["intensity"],
            percentile,
        )
    )
    norm_intensity = matplotlib.colors.PowerNorm(gamma=gamma, vmin=0, vmax=vmax)
    norm_width = matplotlib.colors.Normalize(
        vmin=0,
        vmax=width_limit.to_value(unit_velocity),
    )

    time = a.inputs.time

    if path is None:
        path = default_path / f"level-4-event-gaussians{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=figsize,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    images = []
    for row, component in enumerate(components):

        norm_velocity = matplotlib.colors.Normalize(
            vmin=-component["velocity_limit"],
            vmax=+component["velocity_limit"],
        )

        panels = (
            ("intensity", "intensity", cmap, norm_intensity, unit_intensity),
            ("velocity", "LOS velocity", cmap_velocity, norm_velocity, unit_velocity),
            ("width", "width $\\sigma$", cmap_width, norm_width, unit_velocity),
        )

        for column, (key, title, cmap_panel, norm, unit) in enumerate(panels):
            ax = axs[row, column]
            image = ax.imshow(
                component[key][0],
                origin="lower",
                extent=extent,
                cmap=cmap_panel,
                norm=norm,
                aspect="equal",
            )
            images.append((image, component[key]))
            ax.set_title(f"{component['name']} {title}")
            fig.colorbar(
                image,
                ax=ax,
                label=f"{unit:latex_inline}",
                fraction=0.05,
                pad=0.03,
            )

    axs[0, 0].set_xlim(extent[0], extent[1])
    axs[0, 0].set_ylim(extent[2], extent[3])

    for ax in axs[:, 0]:
        ax.set_ylabel(f"helioprojective $y$ ({unit_position:latex_inline})")
    for ax in axs[-1, :]:
        ax.set_xlabel(f"helioprojective $x$ ({unit_position:latex_inline})")

    if scale_bar is not None:
        megameters_per_arcsec = _megameters_per_arcsec(time[{axis_time: 0}].ndarray)
        length = (scale_bar.to_value(u.Mm) / megameters_per_arcsec * u.arcsec).to_value(
            unit_position
        )
        for ax in axs.flat:
            _scale_bar(ax, length=length, label=f"{scale_bar.to_value(u.Mm):g} Mm")

    def _title(index_time: int) -> str:
        result = f"{label_line} fitted with two Gaussians"
        if not timestamp:
            return result
        t = time[{axis_time: index_time}].ndarray
        return f"{result}, {t.strftime('%Y-%m-%d %H:%M:%S UTC')}"

    text = fig.suptitle(_title(0))

    # Held still, so that the panels do not shift about from one frame to
    # the next as the numbers on the scales change width.
    fig.canvas.draw()
    fig.set_layout_engine("none")

    def func(index_time: int) -> list[matplotlib.artist.Artist]:

        for image, value in images:
            image.set_data(value[index_time])

        text.set_text(_title(index_time))

        return [*[image for image, _ in images], text]

    if path.suffix == ".gif":
        writer = matplotlib.animation.PillowWriter(fps=fps)
    else:
        writer = matplotlib.animation.FFMpegWriter(
            fps=fps,
            codec="h264",
            extra_args=["-pix_fmt", "yuv420p", "-crf", "18"],
        )

    ani = matplotlib.animation.FuncAnimation(fig=fig, func=func, frames=num_time)
    ani.save(filename=path, writer=writer, dpi=dpi)

    plt.close(fig)

    return path


def level_4_event_contours(
    path_data: None | pathlib.Path = None,
    center: None | na.Cartesian2dVectorArray = None,
    radius: u.Quantity = radius_event_default,
    num_frames_dropped: int = 1,
    label_line: str = "O V 630",
    width_min: u.Quantity = width_thermal_o_v,
    width_max: u.Quantity = 150 * u.km / u.s,
    num_passes: int = 10,
    wavelength_aia: u.Quantity = 193 * u.AA,
    percentile: float = 99.99,
    gamma: float = 0.5,
    levels_velocity: u.Quantity = [-100, 100] * u.km / u.s,
    levels_width: u.Quantity = [30, 50] * u.km / u.s,
    colors_velocity: tuple[str, str] = ("dodgerblue", "red"),
    colors_width: tuple[str, ...] = ("gold", "orangered"),
    size_filter: int = 3,
    scale_bar: None | u.Quantity = 10 * u.Mm,
    cmap: str = "gray",
    figsize: tuple[float, float] = figsize_default,
    dpi: float = 150,
    fps: int = 5,
    timestamp: bool = True,
    suffix: str = ".mp4",
    path: None | pathlib.Path = None,
) -> pathlib.Path:
    r"""
    The second Gaussian of :func:`level_4_event_gaussians` drawn as contours
    on AIA, as a loop over the flight.

    Two panels of the same AIA channel, interpolated onto each ESIS frame
    as in :func:`level_4_event`: on the left, the contours of the Doppler
    shift of the second Gaussian, which outline the fast flows toward us
    and away; on the right, the contours of its width, which outline where
    the second Gaussian is broad. What the corona above the flows is doing
    can then be read off beneath them.

    Parameters
    ----------
    path_data
        The directory of Level-4 files to read.
        If :obj:`None`, the directory the local copies are kept in.
    center
        The place on the sky to look at.
        If :obj:`None`, :obj:`center_event_default`.
    radius
        The half width of the region shown.
    num_frames_dropped
        How many frames to leave off the end of the flight, see
        :func:`level_4_event_gaussians`.
    label_line
        The line whose second Gaussian is drawn.
    width_min
        The narrowest a fitted Gaussian may be.
    width_max
        The broadest a fitted Gaussian may be.
    num_passes
        The most passes to make fitting each pair again from the pairs of
        its neighbors, see :func:`level_4_event_gaussians`.
    wavelength_aia
        The AIA channel beneath the contours. By default :math:`193\,\text{\AA}`,
        which is formed in the corona at about a million and a half kelvin,
        far above anything ESIS saw; :math:`304\,\text{\AA}` is the He II
        line of the chromosphere and the transition region, near the
        temperature of the ESIS lines themselves. The channel is in the
        name of the file.
    percentile
        The percentile of AIA placed at the top of its brightness scale.
    gamma
        The power AIA is raised to before it is turned into a brightness, as
        in :func:`level_4_event`.
    levels_velocity
        The Doppler shifts to draw a contour at.
    levels_width
        The widths to draw a contour at, as standard deviations.
    colors_velocity
        The color of the contours of a blueshift and of a redshift.
    colors_width
        The color of each contour of the width, in the order of
        `levels_width`.
    size_filter
        The number of cells across the median filter each map is smoothed
        with before it is contoured, or one for none.

        The second Gaussian goes its own way in scattered single cells,
        and each would get a small ring of its own; a three by three median
        removes them and leaves the jets, which are many cells across.
    scale_bar
        The length of the bar drawn in the upper left of both panels, or
        :obj:`None` for none, as in :func:`level_4_event_gaussians`.
    cmap
        The colormap of AIA.
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution of the saved animation in dots per inch.
    fps
        The number of frames per second in the saved animation.
    timestamp
        Whether to write the time of each exposure in the title.
    suffix
        The file type of the animation, either ``".mp4"`` or ``".gif"``.
        Ignored if `path` is given.
    path
        The location to save the animation.
        If :obj:`None`, it is saved alongside the other figures.

    Notes
    -----
    The contours are placed by the coordinates of the Level-4 product and
    AIA by its own, so they line up only as well as the two were
    coaligned.
    """
    if path_data is None:
        path_data = path_level_4_default.parent

    if center is None:
        center = center_event_default

    a = esis.data.Level_4.from_fits(path_data)

    axis_time = a.axis_time
    unit_velocity = u.km / u.s
    unit_position = na.unit(a.inputs.position.x)

    _, second, _ = _gaussians(
        a=a,
        label_line=label_line,
        center=center,
        radius=radius,
        width_min=width_min,
        width_max=width_max,
        num_passes=num_passes,
        unit_velocity=unit_velocity,
    )

    num_time = second.shape[0] - num_frames_dropped
    size = (1, size_filter, size_filter)
    maps = dict(
        velocity=scipy.ndimage.median_filter(second[:num_time, ..., 1], size=size),
        width=scipy.ndimage.median_filter(second[:num_time, ..., 2], size=size),
    )

    index, _ = _crop_esis(a, center, radius)
    x, y = _centers(a, index)
    x = x.to_value(unit_position)
    y = y.to_value(unit_position)

    # A little wider than the region, since the ESIS cells on its edges
    # reach past it by up to a cell, three quarters of an arcsecond, and
    # AIA cut to the region itself leaves a strip of nothing down the side.
    images_aia, extent_aia, _ = _aia(
        a=a,
        wavelength=wavelength_aia,
        center=center,
        radius=radius + 1 * u.arcsec,
        interpolate=True,
    )
    images_aia = images_aia[:num_time]
    norm_aia = matplotlib.colors.PowerNorm(
        gamma=gamma,
        vmin=0,
        vmax=float(np.nanpercentile(images_aia, percentile)),
    )

    _, extent = _crop_esis(a, center, radius)

    contours = dict(
        velocity=dict(
            levels=levels_velocity.to_value(unit_velocity),
            colors=[
                colors_velocity[0] if level < 0 else colors_velocity[1]
                for level in levels_velocity.to_value(unit_velocity)
            ],
            labels=[
                f"{level:+g} {unit_velocity:latex_inline}"
                for level in levels_velocity.to_value(unit_velocity)
            ],
            title=f"{label_line} second component shift",
        ),
        width=dict(
            levels=levels_width.to_value(unit_velocity),
            colors=list(colors_width[: len(levels_width)]),
            labels=[
                f"$\\sigma$ = {level:g} {unit_velocity:latex_inline}"
                for level in levels_width.to_value(unit_velocity)
            ],
            title=f"{label_line} second component width",
        ),
    )

    time = a.inputs.time
    name_aia = f"AIA {wavelength_aia.to_value(u.AA):.0f}"

    if path is None:
        channel = f"{wavelength_aia.to_value(u.AA):.0f}"
        path = default_path / f"level-4-event-contours-{channel}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figsize,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    pictures = []
    for ax, key in zip(axs, contours):
        pictures.append(
            ax.imshow(
                images_aia[0],
                origin="lower",
                extent=extent_aia,
                cmap=cmap,
                norm=norm_aia,
                aspect="equal",
            )
        )
        ax.set_title(f"{name_aia} and the {contours[key]['title']}")
        ax.set_xlabel(f"helioprojective $x$ ({unit_position:latex_inline})")
        ax.legend(
            handles=[
                matplotlib.lines.Line2D([], [], color=color, label=label)
                for color, label in zip(
                    contours[key]["colors"],
                    contours[key]["labels"],
                )
            ],
            loc="upper right",
            fontsize="small",
            framealpha=0.85,
        )
    axs[0].set_ylabel(f"helioprojective $y$ ({unit_position:latex_inline})")

    # Clipped to the ESIS region, which the AIA crop reaches a little past.
    axs[0].set_xlim(extent[0], extent[1])
    axs[0].set_ylim(extent[2], extent[3])

    if scale_bar is not None:
        megameters_per_arcsec = _megameters_per_arcsec(time[{axis_time: 0}].ndarray)
        length = (scale_bar.to_value(u.Mm) / megameters_per_arcsec * u.arcsec).to_value(
            unit_position
        )
        for ax in axs:
            _scale_bar(ax, length=length, label=f"{scale_bar.to_value(u.Mm):g} Mm")

    def _draw(index_time: int) -> list[matplotlib.artist.Artist]:
        """The contours of one frame."""
        return [
            ax.contour(
                x,
                y,
                maps[key][index_time],
                levels=contours[key]["levels"],
                colors=contours[key]["colors"],
                linewidths=1.5,
            )
            for ax, key in zip(axs, contours)
        ]

    lines = _draw(0)

    def _title(index_time: int) -> str:
        result = f"{name_aia} with the {label_line} flows"
        if not timestamp:
            return result
        t = time[{axis_time: index_time}].ndarray
        return f"{result}, {t.strftime('%Y-%m-%d %H:%M:%S UTC')}"

    text = fig.suptitle(_title(0))

    # Held still, so that the panels do not shift about from one frame to
    # the next.
    fig.canvas.draw()
    fig.set_layout_engine("none")

    def func(index_time: int) -> list[matplotlib.artist.Artist]:

        for picture in pictures:
            picture.set_data(images_aia[index_time])

        # A contour cannot be given new data, so each frame's are drawn
        # afresh in place of the last frame's.
        for line in lines:
            line.remove()
        lines[:] = _draw(index_time)

        text.set_text(_title(index_time))

        return [*pictures, *lines, text]

    if path.suffix == ".gif":
        writer = matplotlib.animation.PillowWriter(fps=fps)
    else:
        writer = matplotlib.animation.FFMpegWriter(
            fps=fps,
            codec="h264",
            extra_args=["-pix_fmt", "yuv420p", "-crf", "18"],
        )

    ani = matplotlib.animation.FuncAnimation(fig=fig, func=func, frames=num_time)
    ani.save(filename=path, writer=writer, dpi=dpi)

    plt.close(fig)

    return path


def _centers(
    a: "esis.data.Level_4",
    index: dict[str, slice],
) -> tuple[u.Quantity, u.Quantity]:
    """
    The middles of the cells of a crop.

    Parameters
    ----------
    a
        The Level-4 product the crop is of.
    index
        The crop.
    """
    result = []
    for axis, vertex in (
        (a.axis_x, a.inputs.position.x.ndarray),
        (a.axis_y, a.inputs.position.y.ndarray),
    ):
        i = index[axis]
        result.append((vertex[i.start : i.stop] + vertex[i.start + 1 : i.stop + 1]) / 2)
    return tuple(result)


def _track(
    a: "esis.data.Level_4",
    label_line: str,
    center: na.Cartesian2dVectorArray,
    radius: u.Quantity,
    percentile_bright: float,
) -> dict[str, np.ndarray]:
    r"""
    The fastest plasma of a line in every frame, both ways, and where it is.

    In each frame the region is searched for the place with the most negative
    Doppler shift and the place with the most positive one, and both are
    reported along with where they are and how bright the line is there. A
    place is followed rather than held still, so what these say is what
    happened to the flow rather than what happened at a point the flow may
    have left.

    Only the brighter part of the region is searched: the median of a faint
    profile is mostly noise, and the largest velocities in a map are
    otherwise always found where there is the least light to measure them
    with.

    Parameters
    ----------
    a
        The Level-4 product to search.
    label_line
        The line to search in.
    center
        The middle of the region to search.
    radius
        The half width of the region to search.
    percentile_bright
        The percentile of the intensity below which a place is too faint for
        its Doppler shift to be believed.
    """
    index_line = list(a.label_line).index(label_line)
    index, _ = _crop_esis(a, center, radius)

    axis = a.axis_wavelength
    radiance = a.outputs[a.window(index_line) | index]

    axes = (a.axis_time, a.axis_y, a.axis_x)
    velocity = (
        na.pdf.median(x=a.velocity, f=radiance, axis=axis)
        .transpose(axes=axes)
        .ndarray.to_value(u.km / u.s)
    )
    intensity = (
        (radiance.sum(axis) * _width(a, index_line)).transpose(axes=axes).ndarray.value
    )

    x, y = _centers(a, index)

    keys = ("x", "y", "v", "i")
    result = {f"{k}_{n}": [] for k in keys for n in ("blue", "red")}

    for k in range(velocity.shape[0]):
        bright = intensity[k] > np.nanpercentile(intensity[k], percentile_bright)
        v = np.where(bright, velocity[k], np.nan)

        for name, place in (
            ("blue", np.unravel_index(np.nanargmin(v), v.shape)),
            ("red", np.unravel_index(np.nanargmax(v), v.shape)),
        ):
            result[f"x_{name}"].append(x[place[1]].value)
            result[f"y_{name}"].append(y[place[0]].value)
            result[f"v_{name}"].append(velocity[k][place])
            result[f"i_{name}"].append(intensity[k][place])

    return {k: np.array(result[k]) for k in result}


def level_4_event_history(
    path_data: None | pathlib.Path = None,
    time_reference: str = "2019-09-30T18:06:46",
    num_frames_dropped: int = 1,
    num_frames_unscaled: int = 1,
    label_line: str = "O V 630",
    center: None | na.Cartesian2dVectorArray = None,
    radius: None | u.Quantity = None,
    radius_image: u.Quantity = radius_event_default,
    offset_image: u.Quantity = offset_event_default,
    percentile_bright: float = 75,
    percentile: float = 99.5,
    gamma: float = 0.5,
    velocity_limit: u.Quantity = 40 * u.km / u.s,
    unit_intensity: u.UnitBase = u.erg / (u.s * u.cm**2 * u.deg**2),
    normalize: bool = False,
    animated: bool = False,
    curves: bool = True,
    marks: bool = True,
    profiles: bool = False,
    cmap: str = "gray",
    cmap_velocity: str = "RdBu_r",
    figsize: tuple[float, float] = figsize_default,
    dpi: float = 150,
    fps: int = 5,
    suffix: None | str = None,
    path: None | pathlib.Path = None,
) -> pathlib.Path:
    r"""
    How fast the event was going, in every line, against time.

    In each frame the region around the event is searched for the place where
    the plasma is coming toward us fastest and the place where it is going
    away fastest, in each line separately. The lower panel is those two
    speeds and the upper panel is how bright the line is where they were
    found, so the two panels describe the same places throughout.

    The places are followed rather than held still, which is the only way to
    ask what happened to the flow: the blue-shifted side of this event
    travels some eleven arcseconds over four minutes, and a fixed point it
    has left reads as plasma coming to rest when the plasma has only gone
    somewhere else.

    The panels on the right are one line at one moment, with the two places
    it was followed to marked, and with the region searched drawn on them:
    a bluer place outside that region is not a place the search could have
    found.

    Parameters
    ----------
    path_data
        The directory of Level-4 files to read.
        If :obj:`None`, the directory the local copies are kept in.
    time_reference
        The moment the panels on the right are of, when not animated.
    num_frames_dropped
        How many frames to leave off the end of the flight.

        By the last frames the rocket is low and the signal has fallen to a
        tenth of what it was at the top, and the search stops finding plasma:
        the extreme runs out to the end of the spectral window, which is a
        report on the noise rather than on the Sun.
    num_frames_unscaled
        How many frames at the end to draw without letting them set the
        velocity scale.

        The same decline, one frame earlier: the last frame kept is worth
        showing but is already running away, and a scale wide enough to hold
        it leaves everything before it squashed flat. It is drawn, and the
        limits are taken from the frames before it.
    label_line
        The line shown on the right, and whose places are marked.
    center
        The middle of the region to search.
        If :obj:`None`, :obj:`position_event_e`.
    radius
        The half width of the region to search, about `center`.

        If :obj:`None`, the whole of what is shown is searched, which is the
        only setting under which a place visibly bluer than the marked one
        cannot appear in the panel. A number here searches a box of that half
        width instead, which keeps the extremes to one event at the cost of
        being able to find a faster one just outside it.
    radius_image
        The half width of the region shown on the right, and the region
        searched when `radius` is :obj:`None`.
    offset_image
        How far north of `center` to put the region shown on the right.

        The event does not stay where it was found: the blue-shifted side
        travels north through the flight, so a region centered on where it
        started leaves it against the top edge by the end. Only what is shown
        is moved, not where the places were looked for.
    percentile_bright
        The percentile of the intensity below which a place is too faint for
        its Doppler shift to be believed.
    percentile
        The percentile of the intensity placed at the top of the brightness
        scale of the panel on the right.
    gamma
        The power the intensity is raised to before it is turned into a
        brightness. One half is the square root, which is the usual way of
        showing a quiet Sun beside something much brighter than it: the event
        is an order of magnitude above the network around it, and on a linear
        scale everything but the event is nearly black. One leaves the
        brightness proportional to the intensity.
    velocity_limit
        The Doppler velocity at each end of the colormap on the right.
    unit_intensity
        The unit to draw the intensities in, see :func:`_energy_photon`.
    normalize
        Whether to divide each line by what it was doing over the whole field
        at the same moment, which costs the intensity its units and leaves a
        ratio.

        Off by default, so that the panel is in the units the product is in
        and the numbers on it are the radiance that was measured. The price
        is that the signal rises and falls through the flight as the rocket
        climbs out of the atmosphere and falls back into it, by a factor of
        two on the way up and a factor of ten on the way down, and that
        envelope is a large part of what these curves do with time. Turn this
        on to divide it out and see how the places behaved against the rest
        of the Sun instead.
    animated
        Whether to save an animation instead of a still.

        The panels on the right then step through the flight and a cursor
        runs across the curves, so that the shape of the event and the
        numbers taken from it are seen at the same moment.
    curves
        Whether to show the curves.

        Hidden rather than never drawn, and hidden only once the layout has
        been settled, so that the two panels of the line itself stay exactly
        where they are in the full figure. One version can then follow the
        other on a slide without the images appearing to move.
    marks
        Whether to mark the two followed places on the panels.
    profiles
        Whether to show the profile of the line at each followed place.

        A column is added between the curves and the images, holding the
        profile at the place of the largest blueshift above the profile at
        the place of the largest redshift, each beside the median profile of
        the whole field at the same moment. Each is averaged over the three
        by three cells around its place, since a single cell is noisy, and
        drawn as a fraction of its own peak: on one scale for the whole
        flight the faint frames and the field are flat against the bottom,
        and it is the shape of a profile, its wings, that the panel is for.
        How bright each place is, the intensity curves already say. The
        median shift the curves report
        says nothing about wings, and wings far out on both sides of a line
        are what an explosive event is recognized by.
    cmap
        The colormap of the intensity panel.
    cmap_velocity
        The colormap of the Doppler panel.
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution of the saved figure in dots per inch.
    fps
        The number of frames per second, if `animated`.
    suffix
        The file type.
        If :obj:`None`, ``".mp4"`` when `animated` and ``".svg"`` otherwise.
    path
        The location to save the figure.
        If :obj:`None`, it is saved alongside the other figures.

    Notes
    -----
    An extreme is bounded below by the noise: it is the largest of many
    numbers, so it can never come out at zero however quiet the Sun is. What
    these curves can say is that one line is faster than another, or that a
    line is faster now than it was; they cannot say that nothing is
    happening.

    The intensities are drawn on a logarithmic scale, since the lines differ
    in brightness by more than an order of magnitude and on a linear scale
    the faint ones would be flat against the bottom of the panel.
    """
    if path_data is None:
        path_data = path_level_4_default.parent

    if center is None:
        center = position_event_e

    if suffix is None:
        suffix = ".mp4" if animated else ".svg"

    a = esis.data.Level_4.from_fits(path_data)

    axis_time = a.axis_time
    axis_wavelength = a.axis_wavelength

    time = astropy.time.Time(np.ravel(a.inputs.time.ndarray))
    num_time = time.size - num_frames_dropped
    time = time[:num_time]
    index_time = int(
        np.argmin(np.abs((time - astropy.time.Time(time_reference)).to_value(u.s)))
    )

    center_image = na.Cartesian2dVectorArray(x=center.x, y=center.y + offset_image)

    # Searching the whole of what is shown is the only way the panels cannot
    # contradict themselves, since a place outside the search is one the mark
    # can never be put on however blue it looks.
    if radius is None:
        center_search, radius_search = center_image, radius_image
    else:
        center_search, radius_search = center, radius

    tracked = {
        label: {
            k: v[:num_time]
            for k, v in _track(
                a=a,
                label_line=label,
                center=center_search,
                radius=radius_search,
                percentile_bright=percentile_bright,
            ).items()
        }
        for label in a.label_line
    }

    # Seconds from the first exposure, which is a scale a reader can hold in
    # their head where a time of day is not.
    seconds = (time - time[0]).to_value(u.s)

    index_line = list(a.label_line).index(label_line)
    index_image, extent_image = _crop_esis(a, center_image, radius_image)

    radiance = a.outputs[a.window(index_line) | index_image]
    axes_image = (axis_time, a.axis_y, a.axis_x)
    image_intensity = (
        (radiance.sum(axis_wavelength) * _width(a, index_line))
        .transpose(axes=axes_image)
        .ndarray.value
    )[:num_time]
    image_velocity = (
        na.pdf.median(x=a.velocity, f=radiance, axis=axis_wavelength)
        .transpose(axes=axes_image)
        .ndarray.to_value(velocity_limit.unit)
    )[:num_time]

    if path is None:
        stem = "level-4-event-history"
        if not curves:
            stem = f"{stem}-images"
        if profiles:
            stem = f"{stem}-profiles"
        path = default_path / f"{stem}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(
        nrows=2,
        ncols=3 if profiles else 2,
        figsize=figsize,
        constrained_layout=True,
        sharex="col",
        width_ratios=(2.0, 1.0, 1.0) if profiles else (3.0, 1.0),
    )

    # The title of the panel at the top right lands flush against the top of
    # the figure, where a renderer clips it, so the layout is given room for
    # it explicitly rather than being left to work it out.
    fig.get_layout_engine().set(h_pad=0.1)

    # One line only. Five of them is ten curves once each is drawn both
    # ways, which is more than a panel this size can say. The two are told
    # apart by color instead, taken from the ends of the colormap the Doppler
    # panel is drawn in, so that the blue curve is the same blue that means
    # approaching there.
    colormap_extreme = plt.get_cmap(cmap_velocity)
    color_extreme = {
        "blue": colormap_extreme(0.1),
        "red": colormap_extreme(0.9),
    }

    if normalize:
        whole = _intensity(a)
        reference = [
            np.nanmedian(
                whole[{a.axis_line: i}]
                .transpose(axes=(axis_time, a.axis_y, a.axis_x))
                .ndarray.value,
                axis=(1, 2),
            )[:num_time]
            for i in range(a.num_line)
        ]
    else:
        reference = [1] * a.num_line

    extremes = (
        ("blue", "solid", "largest blueshift"),
        ("red", "dashed", "largest redshift"),
    )

    if profiles:
        # The profile at each followed place, frame by frame, and the median
        # profile of the whole field at the same moment.
        axes_cube = (axis_time, a.axis_y, a.axis_x, axis_wavelength)
        index_search, _ = _crop_esis(a, center_search, radius_search)
        radiance_search = (
            a.outputs[a.window(index_line) | index_search]
            .transpose(axes=axes_cube)
            .ndarray.value
        )[:num_time]
        x_search, y_search = _centers(a, index_search)
        x_search = np.asarray(x_search.value)
        y_search = np.asarray(y_search.value)
        profile = {}
        for name, _, _ in extremes:
            result = []
            for k in range(num_time):
                i_x = int(np.argmin(np.abs(x_search - tracked[label_line][f"x_{name}"][k])))
                i_y = int(np.argmin(np.abs(y_search - tracked[label_line][f"y_{name}"][k])))
                cells = radiance_search[
                    k,
                    max(i_y - 1, 0) : i_y + 2,
                    max(i_x - 1, 0) : i_x + 2,
                ]
                spectrum = np.nanmean(cells, axis=(0, 1))
                result.append(spectrum / np.nanmax(spectrum))
            profile[name] = np.array(result)
        median_field = np.nanmedian(
            a.outputs[a.window(index_line)].transpose(axes=axes_cube).ndarray.value[
                :num_time
            ],
            axis=(1, 2),
        )
        median_field = median_field / np.nanmax(median_field, axis=1, keepdims=True)
        # The velocity is the edges of the cells, one set shared by every
        # line, rather than a value per cell of each line's window.
        velocity_edges = a.velocity[
            {k: 0 for k in a.velocity.shape if k != axis_wavelength}
        ]
        edges = np.asarray(velocity_edges.ndarray.to_value(velocity_limit.unit))
        if edges.size != radiance_search.shape[~0] + 1:
            raise ValueError(
                f"{edges.size} velocity edges for {radiance_search.shape[~0]} cells"
            )

    # What the product counts, before any of it is turned into energy.
    unit_counted = na.unit(a.outputs) * na.unit(a.inputs.wavelength)

    for name, linestyle, _ in extremes:

        intensity = tracked[label_line][f"i_{name}"]
        if normalize:
            intensity = intensity / reference[index_line]
        else:
            intensity = (
                intensity * unit_counted * _energy_photon(a, index_line)
            ).to_value(unit_intensity)

        axs[0, 0].plot(
            seconds,
            intensity,
            color=color_extreme[name],
            linestyle=linestyle,
        )
        axs[1, 0].plot(
            seconds,
            tracked[label_line][f"v_{name}"],
            color=color_extreme[name],
            linestyle=linestyle,
        )

    axs[0, 0].set_title(label_line, loc="left")

    axs[0, 0].set_yscale("log")
    if normalize:
        axs[0, 0].set_ylabel("intensity / median of the field")
    else:
        axs[0, 0].set_ylabel(f"intensity ({unit_intensity:latex_inline})")

    # The last frames are drawn but are not allowed to set the scale, since
    # by then the extreme is finding noise and a scale wide enough to hold it
    # leaves everything before it flat.
    if num_frames_unscaled:
        scaled = np.concatenate(
            [
                tracked[label][f"v_{name}"][:-num_frames_unscaled]
                for label in a.label_line
                for name, _, _ in extremes
            ]
        )
        margin = 0.05 * np.ptp(scaled)
        axs[1, 0].set_ylim(scaled.min() - margin, scaled.max() + margin)

    axs[1, 0].axhline(0, color="gray", linewidth=0.8, zorder=0)
    axs[1, 0].set_ylabel(f"extreme LOS velocity ({velocity_limit.unit:latex_inline})")
    axs[1, 0].set_xlabel(f"seconds after {time[0].isot[11:19]} UTC")

    # When animated, the moment the panels on the right are showing.
    cursor = [
        ax.axvline(
            seconds[index_time],
            color="gray",
            linewidth=0.8,
            linestyle="dotted",
            zorder=0,
        )
        for ax in axs[:, 0]
    ]

    picture = axs[0, -1].imshow(
        image_intensity[index_time],
        origin="lower",
        extent=extent_image,
        cmap=cmap,
        norm=matplotlib.colors.PowerNorm(
            gamma=gamma,
            vmin=0,
            vmax=float(np.nanpercentile(image_intensity, percentile)),
        ),
        aspect="equal",
    )
    title = axs[0, -1].set_title(f"{label_line}, {time[index_time].isot[11:19]} UTC")

    image = axs[1, -1].imshow(
        image_velocity[index_time],
        origin="lower",
        extent=extent_image,
        cmap=cmap_velocity,
        vmin=-velocity_limit.to_value(velocity_limit.unit),
        vmax=+velocity_limit.to_value(velocity_limit.unit),
        aspect="equal",
    )
    fig.colorbar(
        image,
        ax=axs[1, -1],
        location="bottom",
        fraction=0.06,
        pad=0.02,
        label=f"LOS velocity ({velocity_limit.unit:latex_inline})",
    )

    unit_position = na.unit(a.inputs.position.x)
    axs[1, -1].set_xlabel(f"helioprojective $x$ ({unit_position:latex_inline})")
    for ax in axs[:, -1]:
        ax.set_ylabel(f"helioprojective $y$ ({unit_position:latex_inline})")

    # The same axes in megameters, so the size of the event can be read off
    # without anybody having to remember what an arcsecond is worth. A twin
    # rather than a replacement, since the arcseconds are what the other
    # figures are in.
    scale = _megameters_per_arcsec(time[index_time])
    functions = (lambda v: v * scale, lambda v: v / scale)
    for ax in axs[:, -1]:
        ax.secondary_yaxis("right", functions=functions).set_ylabel("$y$ (Mm)")
    axs[1, -1].secondary_xaxis("top", functions=functions).set_xlabel("$x$ (Mm)")

    # Said once, rather than left to whatever was drawn last.
    for ax in axs[:, -1]:
        ax.set_xlim(extent_image[0], extent_image[1])
        ax.set_ylim(extent_image[2], extent_image[3])

    # Outlined, since a mark has to be found against black in one panel and
    # against white in the other.
    outline = [matplotlib.patheffects.withStroke(linewidth=3.5, foreground="white")]

    marker = {"blue": "o", "red": "s"}
    marker_name = {"blue": "circle", "red": "square"}
    follower = (
        {}
        if not marks
        else {
            name: [
                ax.plot(
                    tracked[label_line][f"x_{name}"][index_time],
                    tracked[label_line][f"y_{name}"][index_time],
                    marker=marker[name],
                    markerfacecolor="none",
                    markeredgecolor="black",
                    markersize=10,
                    markeredgewidth=1.8,
                    linestyle="none",
                    path_effects=outline,
                )[0]
                for ax in axs[:, -1]
            ]
            for name, _, _ in extremes
        }
    )

    axs[1, 0].legend(
        handles=[
            matplotlib.lines.Line2D(
                [],
                [],
                color=color_extreme[name],
                linestyle=linestyle,
                label=description,
            )
            for name, linestyle, description in extremes
        ],
        loc="upper left",
        fontsize="small",
    )

    # The marks are explained on the panel they are drawn on, rather than in
    # the legend above: a handle carrying a line style and a mark at once is
    # illegible at this size, the dashes and the mark running together.
    if marks:
        axs[0, -1].legend(
            handles=[
                matplotlib.lines.Line2D(
                    [],
                    [],
                    markerfacecolor="none",
                    markeredgecolor="black",
                    linestyle="none",
                    marker=marker[name],
                    markersize=7,
                    markeredgewidth=1.8,
                    label=description,
                )
                for name, _, description in extremes
            ],
            loc="upper right",
            fontsize="x-small",
            framealpha=0.85,
            handletextpad=0.4,
            borderpad=0.4,
        )

    stairs_place = {}
    stairs_field = []
    if profiles:
        for ax, (name, linestyle, description) in zip(axs[:, 1], extremes):
            stairs_field.append(
                ax.stairs(
                    median_field[index_time],
                    edges,
                    color="gray",
                    linewidth=1.2,
                    label="median of the field",
                )
            )
            stairs_place[name] = ax.stairs(
                profile[name][index_time],
                edges,
                color=color_extreme[name],
                linestyle=linestyle,
                linewidth=1.8,
                label=f"at the {description}",
            )
            ax.axvline(0, color="gray", linewidth=0.8, zorder=0)
            # Room above the peak for the legend, which would otherwise sit
            # on the profile.
            ax.set_ylim(0, 1.3)
            ax.set_ylabel("relative intensity")
            ax.set_title(f"{label_line} at the {marker_name[name]}", fontsize="medium")
            ax.legend(loc="upper left", fontsize="x-small", framealpha=0.85)
        axs[1, 1].set_xlim(edges[0], edges[-1])
        axs[1, 1].set_xlabel(f"velocity ({velocity_limit.unit:latex_inline})")

    # The layout is worked out once and then frozen, so that the panels do
    # not shift about from one frame to the next as the labels change width,
    # and so that hiding the curves below leaves everything else where it is.
    fig.canvas.draw()
    fig.set_layout_engine("none")

    if not curves:
        for ax in axs[:, 0]:
            ax.set_visible(False)

    if not animated:
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return path

    def func(index: int) -> list[matplotlib.artist.Artist]:

        picture.set_data(image_intensity[index])
        image.set_data(image_velocity[index])

        for name in follower:
            for artist in follower[name]:
                artist.set_data(
                    [tracked[label_line][f"x_{name}"][index]],
                    [tracked[label_line][f"y_{name}"][index]],
                )

        for artist in cursor:
            artist.set_xdata([seconds[index], seconds[index]])
        for name in stairs_place:
            stairs_place[name].set_data(values=profile[name][index])
        for artist in stairs_field:
            artist.set_data(values=median_field[index])

        title.set_text(f"{label_line}, {time[index].isot[11:19]} UTC")

        return [
            picture,
            image,
            *[artist for name in follower for artist in follower[name]],
            *cursor,
            *stairs_place.values(),
            *stairs_field,
            title,
        ]

    if path.suffix == ".gif":
        writer = matplotlib.animation.PillowWriter(fps=fps)
    else:
        writer = matplotlib.animation.FFMpegWriter(
            fps=fps,
            codec="h264",
            extra_args=["-pix_fmt", "yuv420p", "-crf", "18"],
        )

    ani = matplotlib.animation.FuncAnimation(fig=fig, func=func, frames=num_time)
    ani.save(filename=path, writer=writer, dpi=dpi)

    plt.close(fig)

    return path


def level_4_event_motion(
    path_data: None | pathlib.Path = None,
    time_reference: str = "2019-09-30T18:06:46",
    num_frames_dropped: int = 1,
    num_frames_unscaled: int = 1,
    label_line: str = "O V 630",
    center: None | na.Cartesian2dVectorArray = None,
    radius: None | u.Quantity = None,
    radius_image: u.Quantity = radius_event_default,
    offset_image: u.Quantity = offset_event_default,
    percentile_bright: float = 75,
    percentile: float = 99.5,
    gamma: float = 0.5,
    velocity_limit: u.Quantity = 40 * u.km / u.s,
    speed_limit: None | u.Quantity = 300 * u.km / u.s,
    second: bool = False,
    width_min: u.Quantity = width_thermal_o_v,
    width_max: u.Quantity = 150 * u.km / u.s,
    num_passes: int = 10,
    animated: bool = False,
    cmap: str = "gray",
    cmap_velocity: str = "RdBu_r",
    figsize: tuple[float, float] = figsize_default,
    dpi: float = 150,
    fps: int = 5,
    suffix: None | str = None,
    path: None | pathlib.Path = None,
) -> pathlib.Path:
    r"""
    How fast the event moved, across the sky as well as along the line of
    sight.

    The companion to :func:`level_4_event_history`, which asks how fast the
    plasma was going toward us. The same two places are followed, the ones
    where the line is most blue-shifted and most red-shifted, and here what
    is asked is how fast those places themselves travelled across the sky,
    and what the two speeds come to together.

    A panel for each direction, and in each of them both speeds at once. The
    speed across the sky is nearly all of the total, so the two curves lie
    almost on top of one another and the gap between them, shaded, is
    everything the Doppler shift contributes.

    On the right the same field is drawn twice, its brightness and its
    Doppler shift, with the two places marked on both, so that what the
    numbers on the left are measurements of can be seen directly.

    Parameters
    ----------
    path_data
        The directory of Level-4 files to read.
        If :obj:`None`, the directory the local copies are kept in.
    time_reference
        The moment the panels on the right are of, when not animated.
    num_frames_dropped
        How many frames to leave off the end of the flight, see
        :func:`level_4_event_history`.
    num_frames_unscaled
        How many frames at the end to draw without letting them set the
        scale.
    label_line
        The line shown on the right, and whose places are marked.
    center
        The middle of the region to search.
        If :obj:`None`, :obj:`position_event_e`.
    radius
        The half width of the region to search.
        If :obj:`None`, the whole of what is shown.
    radius_image
        The half width of the region shown on the right.
    offset_image
        How far north of `center` to put the region shown on the right.
    percentile_bright
        The percentile of the intensity below which a place is too faint for
        its Doppler shift to be believed.
    percentile
        The percentile of the intensity placed at the top of the brightness
        scale of the panel on the right.
    gamma
        The power the intensity is raised to before it is turned into a
        brightness. One half is the square root, which is the usual way of
        showing a quiet Sun beside something much brighter than it: the event
        is an order of magnitude above the network around it, and on a linear
        scale everything but the event is nearly black. One leaves the
        brightness proportional to the intensity.
    velocity_limit
        The Doppler velocity at each end of the colormap on the right.
    speed_limit
        The speed at the top of the two panels on the left.

        The curves are mostly quiet and occasionally leave the plot
        altogether, in the frames where the place being followed stops being
        one feature and becomes another. A scale wide enough to hold those
        excursions leaves the quiet stretches, which are the measurement,
        flat against the bottom. They are drawn and clipped instead. If
        :obj:`None`, the scale is taken from the data.
    second
        Whether to use the Doppler shift of the second of two Gaussians
        fitted to each profile of `label_line`, see
        :func:`level_4_event_gaussians`, rather than the median of the
        profile, in the Doppler panel and in the total speed.

        The median of a profile with a fast component in one wing moves
        only a fraction of the way toward it, by however much of the line
        the component carries, so it says that something is moving but not
        how fast. The second Gaussian is the moving plasma by itself.

        The places followed are the same either way, the ones where the
        median is most shifted, so that they are the places
        :func:`level_4_event_history` follows too. Searching the second
        Gaussian for its largest shift instead finds the ragged edges of
        the jets, where the fits have the least light to work with.
    width_min
        The narrowest a fitted Gaussian may be, if `second`.
    width_max
        The broadest a fitted Gaussian may be, if `second`.
    num_passes
        The most passes to make fitting each pair again from the pairs of
        its neighbors, if `second`, see :func:`level_4_event_gaussians`.
    animated
        Whether to save an animation instead of a still.
    cmap
        The colormap of the intensity panel.
    cmap_velocity
        The colormap of the Doppler panel.
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution of the saved figure in dots per inch.
    fps
        The number of frames per second, if `animated`.
    suffix
        The file type.
        If :obj:`None`, ``".mp4"`` when `animated` and ``".svg"`` otherwise.
    path
        The location to save the figure.
        If :obj:`None`, it is saved alongside the other figures.

    Notes
    -----
    The place followed is the one where the line is most shifted anywhere in
    the field, found afresh in every frame and with nothing tying it to where
    it was in the frame before. That is the right definition for a Doppler
    shift and the wrong one for a speed across the sky: when the fastest
    place stops being one feature and starts being another, the position
    jumps, and a jump divided by ten seconds is a very large number. A cell
    is half a megameter, so one cell of jump is already fifty kilometers per
    second, and the field is seventy-eight cells across.

    These speeds are therefore a lower bound on nothing and an upper bound on
    nothing; they say how far apart the two extremes were from one frame to
    the next. Read them together with the running difference beside them,
    which shows whether anything actually moved.
    """
    if path_data is None:
        path_data = path_level_4_default.parent

    if center is None:
        center = position_event_e

    if suffix is None:
        suffix = ".mp4" if animated else ".svg"

    a = esis.data.Level_4.from_fits(path_data)

    axis_time = a.axis_time
    axis_wavelength = a.axis_wavelength

    time = astropy.time.Time(np.ravel(a.inputs.time.ndarray))
    num_time = time.size - num_frames_dropped
    time = time[:num_time]
    index_time = int(
        np.argmin(np.abs((time - astropy.time.Time(time_reference)).to_value(u.s)))
    )

    center_image = na.Cartesian2dVectorArray(x=center.x, y=center.y + offset_image)

    if radius is None:
        center_search, radius_search = center_image, radius_image
    else:
        center_search, radius_search = center, radius

    tracked = {
        label: {
            k: v[:num_time]
            for k, v in _track(
                a=a,
                label_line=label,
                center=center_search,
                radius=radius_search,
                percentile_bright=percentile_bright,
            ).items()
        }
        for label in a.label_line
    }

    if second:
        # The same places, but the Doppler shift of the second Gaussian at
        # each of them in place of the median of the line there.
        _, second_search, _ = _gaussians(
            a=a,
            label_line=label_line,
            center=center_search,
            radius=radius_search,
            width_min=width_min,
            width_max=width_max,
            num_passes=num_passes,
        )
        index_search, _ = _crop_esis(a, center_search, radius_search)
        x_search, y_search = _centers(a, index_search)
        for name in ("blue", "red"):
            i_x = np.argmin(
                np.abs(x_search.value[:, np.newaxis] - tracked[label_line][f"x_{name}"]),
                axis=0,
            )
            i_y = np.argmin(
                np.abs(y_search.value[:, np.newaxis] - tracked[label_line][f"y_{name}"]),
                axis=0,
            )
            tracked[label_line][f"v_{name}"] = second_search[
                np.arange(num_time), i_y, i_x, 1
            ]

    seconds = (time - time[0]).to_value(u.s)

    # An arcsecond is a distance only once the Sun's distance is known, and
    # that is a property of the day rather than a constant.
    kilometers_per_arcsec = _megameters_per_arcsec(time[index_time]) * 1e3

    index_line = list(a.label_line).index(label_line)
    index_image, extent_image = _crop_esis(a, center_image, radius_image)

    radiance = a.outputs[a.window(index_line) | index_image]
    axes_image = (axis_time, a.axis_y, a.axis_x)
    image_intensity = (
        (radiance.sum(axis_wavelength) * _width(a, index_line))
        .transpose(axes=axes_image)
        .ndarray.value
    )[:num_time]
    if second:
        _, second_image, _ = _gaussians(
            a=a,
            label_line=label_line,
            center=center_image,
            radius=radius_image,
            width_min=width_min,
            width_max=width_max,
            num_passes=num_passes,
            unit_velocity=velocity_limit.unit,
        )
        image_velocity = second_image[:num_time, ..., 1]
    else:
        image_velocity = (
            na.pdf.median(x=a.velocity, f=radiance, axis=axis_wavelength)
            .transpose(axes=axes_image)
            .ndarray.to_value(velocity_limit.unit)
        )[:num_time]

    # What the Doppler shift in the total is of, where it is not the median
    # of the line that also chose the places.
    label_total = "total, with the second component" if second else "total"

    if path is None:
        stem = "level-4-event-motion"
        if second:
            stem = f"{stem}-second"
        path = default_path / f"{stem}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=figsize,
        constrained_layout=True,
        sharex="col",
        width_ratios=(3.0, 1.0),
    )
    fig.get_layout_engine().set(h_pad=0.1)

    extremes = (
        ("blue", "solid", "largest blueshift"),
        ("red", "dashed", "largest redshift"),
    )

    # Receding above, approaching below, which is the order they are in on
    # the Doppler panel to the right of them.
    axis_extreme = {"red": axs[0, 0], "blue": axs[1, 0]}

    # Taken from the ends of the colormap the Doppler panel is drawn in, so
    # that the red curve is the same red that means receding there. Short of
    # the very ends, which are dark enough to read as black.
    colormap_velocity = plt.get_cmap(cmap_velocity)
    color_extreme = {"blue": colormap_velocity(0.1), "red": colormap_velocity(0.9)}

    speed = {}
    for name, _, description in extremes:

        # The speed across the sky, from how far the place moved between one
        # frame and the next.
        velocity_x = np.gradient(tracked[label_line][f"x_{name}"], seconds)
        velocity_y = np.gradient(tracked[label_line][f"y_{name}"], seconds)
        across = np.hypot(velocity_x, velocity_y) * kilometers_per_arcsec

        # The two put together, which is the speed of the plasma if the place
        # followed is the same piece of plasma throughout.
        total = np.hypot(across, tracked[label_line][f"v_{name}"])

        speed[name] = (across, total)

        ax = axis_extreme[name]
        ax.fill_between(
            seconds,
            across,
            total,
            color=color_extreme[name],
            alpha=0.3,
            linewidth=0,
        )
        ax.plot(seconds, across, color=color_extreme[name], linestyle="solid")
        ax.plot(seconds, total, color="black", linestyle="dashed", linewidth=1)
        ax.set_title(f"{label_line}, {description}", loc="left")

        # On each panel rather than once, since the color of the curve it
        # describes is not the same on both.
        ax.legend(
            handles=[
                matplotlib.lines.Line2D(
                    [],
                    [],
                    color=color_extreme[name],
                    linestyle="solid",
                    label="plane of sky",
                ),
                matplotlib.lines.Line2D(
                    [],
                    [],
                    color="black",
                    linestyle="dashed",
                    linewidth=1,
                    label=label_total,
                ),
            ],
            loc="upper center",
            fontsize="small",
        )

    unit_speed = u.km / u.s

    if speed_limit is not None:
        for ax in axs[:, 0]:
            ax.set_ylim(0, speed_limit.to_value(unit_speed))
    elif num_frames_unscaled:
        # The last frames are drawn but are not allowed to set the scale.
        for name, _, _ in extremes:
            scaled = np.concatenate([v[:-num_frames_unscaled] for v in speed[name]])
            axis_extreme[name].set_ylim(0, scaled.max() * 1.05)
    for ax in axs[:, 0]:
        ax.set_ylabel(f"speed ({unit_speed:latex_inline})")
    axs[1, 0].set_xlabel(f"seconds after {time[0].isot[11:19]} UTC")

    cursor = [
        ax.axvline(
            seconds[index_time],
            color="gray",
            linewidth=0.8,
            linestyle="dotted",
            zorder=0,
        )
        for ax in axs[:, 0]
    ]

    picture = axs[0, 1].imshow(
        image_intensity[index_time],
        origin="lower",
        extent=extent_image,
        cmap=cmap,
        norm=matplotlib.colors.PowerNorm(
            gamma=gamma,
            vmin=0,
            vmax=float(np.nanpercentile(image_intensity, percentile)),
        ),
        aspect="equal",
    )
    title = axs[0, 1].set_title(f"{label_line}, {time[index_time].isot[11:19]} UTC")

    image = axs[1, 1].imshow(
        image_velocity[index_time],
        origin="lower",
        extent=extent_image,
        cmap=cmap_velocity,
        vmin=-velocity_limit.to_value(velocity_limit.unit),
        vmax=+velocity_limit.to_value(velocity_limit.unit),
        aspect="equal",
    )
    label_velocity = f"LOS velocity ({velocity_limit.unit:latex_inline})"
    if second:
        label_velocity = f"second component {label_velocity}"
    fig.colorbar(
        image,
        ax=axs[1, 1],
        location="bottom",
        fraction=0.06,
        pad=0.02,
        label=label_velocity,
    )

    unit_position = na.unit(a.inputs.position.x)
    axs[1, 1].set_xlabel(f"helioprojective $x$ ({unit_position:latex_inline})")
    for ax in axs[:, 1]:
        ax.set_ylabel(f"helioprojective $y$ ({unit_position:latex_inline})")

    scale = _megameters_per_arcsec(time[index_time])
    functions = (lambda v: v * scale, lambda v: v / scale)
    for ax in axs[:, 1]:
        ax.secondary_yaxis("right", functions=functions).set_ylabel("$y$ (Mm)")
    axs[1, 1].secondary_xaxis("top", functions=functions).set_xlabel("$x$ (Mm)")

    for ax in axs[:, 1]:
        ax.set_xlim(extent_image[0], extent_image[1])
        ax.set_ylim(extent_image[2], extent_image[3])

    outline = [matplotlib.patheffects.withStroke(linewidth=3.5, foreground="white")]

    marker = {"blue": "o", "red": "s"}
    follower = {
        name: [
            ax.plot(
                tracked[label_line][f"x_{name}"][index_time],
                tracked[label_line][f"y_{name}"][index_time],
                marker=marker[name],
                markerfacecolor="none",
                markeredgecolor="black",
                markersize=10,
                markeredgewidth=1.8,
                linestyle="none",
                path_effects=outline,
            )[0]
            for ax in axs[:, 1]
        ]
        for name, _, _ in extremes
    }

    axs[0, 1].legend(
        handles=[
            matplotlib.lines.Line2D(
                [],
                [],
                markerfacecolor="none",
                markeredgecolor="black",
                linestyle="none",
                marker=marker[name],
                markersize=7,
                markeredgewidth=1.8,
                label=description,
            )
            for name, _, description in extremes
        ],
        loc="upper right",
        fontsize="x-small",
        framealpha=0.85,
        handletextpad=0.4,
        borderpad=0.4,
    )

    if not animated:
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return path

    fig.canvas.draw()
    fig.set_layout_engine("none")

    def func(index: int) -> list[matplotlib.artist.Artist]:

        picture.set_data(image_intensity[index])
        image.set_data(image_velocity[index])

        for name, _, _ in extremes:
            for artist in follower[name]:
                artist.set_data(
                    [tracked[label_line][f"x_{name}"][index]],
                    [tracked[label_line][f"y_{name}"][index]],
                )

        for artist in cursor:
            artist.set_xdata([seconds[index], seconds[index]])

        title.set_text(f"{label_line}, {time[index].isot[11:19]} UTC")

        return [
            picture,
            image,
            *[artist for name in follower for artist in follower[name]],
            *cursor,
            title,
        ]

    if path.suffix == ".gif":
        writer = matplotlib.animation.PillowWriter(fps=fps)
    else:
        writer = matplotlib.animation.FFMpegWriter(
            fps=fps,
            codec="h264",
            extra_args=["-pix_fmt", "yuv420p", "-crf", "18"],
        )

    ani = matplotlib.animation.FuncAnimation(fig=fig, func=func, frames=num_time)
    ani.save(filename=path, writer=writer, dpi=dpi)

    plt.close(fig)

    return path
