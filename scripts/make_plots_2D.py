"""
Plotting script for pepper histogram output.
Can read both ROOT and hist (.coffea) histograms.

Steered via a plot_config.json file
(https://cms-pepper.docs.cern.ch/examples/bundled_scripts.html#step-8-plotting)

While it is quite flexible, it is not meant as a one-fits-all
solution, but instead should be extended and adapted to the needs
of your analysis.

Structure of the code:
- Actual plotting happens in the `plot_2d` function. If you want to
    change the plot style, this is your first step.
- File I/O and categorization is done in the latter half of the
    script.

"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mplhep
from argparse import ArgumentParser
import tqdm
import hist
import itertools
import traceback
import multiprocessing
from warnings import filterwarnings
import matplotlib.colors as colors

# The pepper class "HistCollection" is used to load hists.json files
from pepper import Config, HistCollection
# Utility methods for working with `hist` histograms
from pepper.hist_utils import group_histogram, replace_missing_systematics

filterwarnings("ignore", "List indexing selection is experimental")
filterwarnings("ignore", "invalid value encountered in divide")
filterwarnings("ignore", "divide by zero encountered in divide")
filterwarnings("ignore", "invalid value encountered in add")
filterwarnings("ignore", "invalid value encountered in double_scalars")

# Use Agg (non-interactive) backend, which plays nicer with headless
# environments
matplotlib.use("Agg")
plt.set_loglevel("error")

# Use the Official CMS Plot Style (TM)
# Requires an up-to-date mplhep version
plt.style.use(mplhep.style.CMS)
cms_color_scheme = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_2d(h, config, ax_x, ax_y, outfile, cutname, exts):
    """
    Plot a 2D histogram.

    This is the method you should modify if you want to change the
    plotting style.

    Parameters
    ----------
    h
        A histogram containing all MC samples, possible data and signals,
        and systematic variations if present, already grouped by process.
    config
        The plotting config.
    ax_x
        The x-axis of the 2D plot.
    ax_y
        The y-axis of the 2D plot.
    outfile
        Output file for the plot.
    """

    # Setup plot labels
    year = config["year"] if "year" in config else None
    luminosity = config["luminosity"] if "luminosity" in config else None
    com_energy = config["com_energy"] if "com_energy" in config else 13
    cmslabel = config["cmslabel"] if "cmslabel" in config else None

    # Prepare a list of histograms to plot
    to_plot = []

    # Add individual processes
    for proc in h.axes["process"]:
        h_proc = h[{"process": proc}]
        if "sys" in h_proc.axes.name:
            h_proc = h_proc[{"sys": "nominal"}]
        # if np.sum(h_proc.values()) > 0:
        to_plot.append((h_proc, proc, proc == "data"))

    # Get summed yields for all background processes
    all_backgrounds = list(config["backgrounds"].keys())
    # Filter only backgrounds actually present in this histogram
    available_bgs = [b for b in all_backgrounds if b in h.axes["process"]]

    if len(available_bgs) > 0:
        h_mc = h.integrate("process", available_bgs)
        if "sys" in h_mc.axes.name:
            h_mc = h_mc[{"sys": "nominal"}]
        to_plot.append((h_mc, "Total_MC", False))

    # Plot the histograms for each process
    for h_proc, proc_label, _is_data in to_plot:
        # Make a plot
        fig, ax = plt.subplots()

        mplhep.hist2dplot(
            h_proc,
            ax=ax,
            cmap="viridis",
            cbar=True,
            cbarextend=True
        )

        ax.set_xlabel(ax_x.label)
        ax.set_ylabel(ax_y.label)

        mplhep.cms.label(
            ax=ax,
            label=cmslabel,
            data=True,
            year=year,
            lumi=f"{luminosity:.1f}",
            com=com_energy
        )

        # Annotations
        ann = f"Process: {proc_label}\n"
        if "cut_labels" in config and cutname in config["cut_labels"]:
            ann += config["cut_labels"][cutname] + "\n"
        ax.annotate(ann, (0.02, 0.96), xycoords="axes fraction",
                    va="top", fontsize="small", color="white")

        fig.align_ylabels()
        plt.tight_layout()

        # Save file
        proc_outfile = f"{outfile}_{proc_label}"
        # Save in all formats specified in the command line arguments
        for ext in exts:
            fig.savefig(proc_outfile + "." + ext)
        plt.close()

    # Make comparison plot
    h_nominal = h[{"sys": "nominal"}] if "sys" in h.axes.name else h

    has_data = "data" in h_nominal.axes["process"]
    # Get summed yields for all background processes
    all_backgrounds = list(config["backgrounds"].keys())

    # Filter only backgrounds actually present in this histogram
    available_bgs = [b for b in all_backgrounds if b in h_nominal.axes["process"]]

    if has_data and len(available_bgs) > 0:
        h_data = h_nominal[{"process": "data"}]
        h_mc = h_nominal.integrate("process", available_bgs)

        data_vals = h_data.values()
        mc_vals = h_mc.values()

        # Initialize ratio array
        ratio_vals = np.full_like(data_vals, np.nan, dtype=float)

        # Calculate ratio when MC values > 0
        mask = mc_vals > 0
        ratio_vals[mask] = data_vals[mask] / mc_vals[mask]

        # If MC is zero and data is not, set a high number
        # This will show up as a hotspot in the ratio plot,
        # you can remove this, and just leave it as np.nan
        # which will show up as transparent.
        mask_hotspot = (mc_vals == 0) & (data_vals > 0)
        ratio_vals[mask_hotspot] = 999.0

        # Create figure for the ratio map
        fig, ax = plt.subplots()

        # Set a diverging colormap centered at 1, with a range from 0 to 2
        norm = colors.TwoSlopeNorm(vcenter=1.0, vmin=0.0, vmax=2.0)

        mesh = mplhep.hist2dplot(
            ratio_vals,
            xbins=ax_x.edges,
            ybins=ax_y.edges,
            ax=ax,
            cmap="bwr",
            cbar=True,
            cbarextend=True,
            norm=norm
        )

        cb = mesh.cbar
        cb.set_label("Data / Pred.", fontsize="large")

        ax.set_xlabel(ax_x.label)
        ax.set_ylabel(ax_y.label)

        mplhep.cms.label(
            ax=ax,
            label=cmslabel,
            data=True,
            year=year,
            lumi=f"{luminosity:.1f}",
            com=com_energy
        )

        # Annotations
        ann = "Data vs. MC\n"
        if "cut_labels" in config and cutname in config["cut_labels"]:
            ann += config["cut_labels"][cutname] + "\n"
        ax.annotate(ann, (0.02, 0.96), xycoords="axes fraction",
                    va="top", fontsize="small", color="black")

        plt.tight_layout()

        # Save the Comparison Map
        ratio_outfile = f"{outfile}_DatavsMC"
        for ext in exts:
            fig.savefig(ratio_outfile + "." + ext)
        plt.close()


def get_category_dict(h, config):
    """
    Build a dictionary of containing definitions of all categories
    we want to plot for a specific histogram, as specified in the
    plot config.

    Modify this if you want different category behaviour.

    Parameters
    ----------
    h
        The histogram to categorize, already grouped by process.
    config
        The plotting config.

    Returns
    -------
        A dictionary containing the category defintions.
    """

    # List of coarse (i.e. category) axes in the histograms
    # except for systematics and processes, we treat those extra
    coarse_axes = [
        ax
        for ax in h.axes
        if (isinstance(ax, hist.axis.StrCategory)
            or isinstance(ax, hist.axis.IntCategory))
        and ax.name != "sys" and ax.name != "process"
    ]

    # Make a dict which, for each category we want to plot, contains a
    # list of which axis values should be included in this category
    category_dict = {}

    # If 'sum_categories' is true in the config, simply make one category
    # called "sum" which sums over all coarse axes in the histogram
    if "sum_categories" in config and config["sum_categories"]:
        category_dict["sum"] = {
            ax.name: list(ax) for ax in coarse_axes
        }

    # If categories are given in the config explicitly, use those
    elif "categories" in config and len(config["categories"]) > 0:
        # Iterate over the categories defined in the config
        for category_label, category_definition \
                in config["categories"].items():

            # Check that at least one of the axes in the histogram
            # has a definition in the category. Otherwise, skip the
            # category
            if any(axname not in h.axes.name
                   for axname in category_definition.keys()):
                print(f"Warning: None of the coarse axes specified for "
                      f"category {category_label} was found in histogram. "
                      f"Ignoring this category")
                continue

            # Copy the category definition for modification
            category_definition_modified = category_definition.copy()

            # Check for axes that are present in the histogram but missing
            # in the category definition in the config
            for ax in coarse_axes:
                if ax.name not in category_definition:
                    # Axis is present in histogram but missing in config
                    # --> sum over this axis
                    category_definition_modified[ax.name] = list(ax)
                    print(f"Warning: Coarse axis {ax.name} found in histogram "
                          f"but not in definition for category "
                          f"{category_label}. Will sum over this axis")

            # Add to the dictionary of categories
            category_dict[category_label] = category_definition_modified

        # If no categories are valid for this histogram (e.g. because it
        # is early in the cutflow), fall back to plotting the sum
        if len(category_dict) == 0:
            category_dict["sum"] = {
                ax.name: list(ax) for ax in coarse_axes
            }

    # If nothing is defined in the config, build a dictionary with all possible
    # combinations of values present in the coarse axes of the histogram
    else:
        # For each coarse_axis, use all values in the histogram, as well as
        # a sum over that axis
        all_axis_values = [
            [*ax, "sum"] for ax in coarse_axes
        ]
        # Make a cartesian product using itertools
        for axis_values in itertools.product(*all_axis_values):
            # Build a dictionary for this category and a label
            category_definition = {}
            label_parts = []
            for ax, axis_value in zip(coarse_axes, axis_values):
                if axis_value == "sum":
                    category_definition[ax.name] = list(ax)
                else:
                    category_definition[ax.name] = axis_value
                label_parts.append(ax.name + "_" + axis_value)
            category_label = "_".join(label_parts)
            category_dict[category_label] = category_definition

    return category_dict


# Set up command line arguments
parser = ArgumentParser(
    description="Plot 2D histograms from previously created histograms")
parser.add_argument(
    "plot_config", help="Path to a configuration file for plotting")
parser.add_argument(
    "histfile", help="Coffea file with a single histogram or a "
    "JSON file containing histogram info. See output of select_events.py")
parser.add_argument(
    "-o", "--outdir", help="Output directory. If not given, output to the "
    "directory where histfile is located")
parser.add_argument(
    "--ext", choices=["pdf", "svg", "png"], help="Output file format",
    action="append")
parser.add_argument(
    "-c", "--cut", type=str, default=None, help="If specified, only "
    "plot a given cut (i.e. only 'Req MET'.)")
parser.add_argument(
    "-j", "--jobs", type=int, default=1, help="Parallelize over given "
    "number of jobs. Default 1")
args = parser.parse_args()

# Set the default output format here
if args.ext is None:
    args.ext = ["pdf"]

# Read the config from json using the pepper Config class
config = Config(args.plot_config)

# From the config, build a dictionary for grouping the datasets in
# the histogram into processes
# This includes BG processes, signal processes, and observed data
groups = {p: b["datasets"] for p, b in config["backgrounds"].items()}
if "signals" in config:
    for p, s in config["signals"].items():
        groups[p] = s["datasets"]
if "data" in config:
    groups["data"] = config["data"]

# Do we plot a collection of hists using a hists.json file, or
# just a single histogram ?
is_hist_json = args.histfile.endswith(".json")
if is_hist_json:
    # Read the info about all histograms from the hists.json
    # by constructing a HistCollection object
    with open(args.histfile) as f:
        hists = HistCollection.from_json(f)
else:
    # Create a HistCollection for only the single histogram
    hists = HistCollection.from_single_hist(args.histfile)


def process_histogram(histkey):

    # Retrieve the cut and the variable for the histogram
    if is_hist_json:
        cutname, variable = histkey
    else:
        # Try to get cut and variable from the hist file name
        filename = os.path.basename(args.histfile).split(".")[0]
        # cutname, variable = "Unknown", filename

        if filename.startswith("Cut "):
            # Old file name syntax (with spaces)
            filename = " ".join(filename.split(" ")[2:])
            filename_parts = filename.split("_")
            cutname = filename_parts[0]
            variable = "_".join(filename_parts[1:])
        elif filename.startswith("Cut_"):
            # New file name syntax (with underscores)
            filename_parts = filename.split("_")
            cutname = filename_parts[2]
            variable = "_".join(filename_parts[3:])
        else:
            # Otherwise give up
            cutname = ""
            variable = ""

    # If "--cut" is given as a command line argument, and
    # the histogram is from a different cut, skip
    if args.cut is not None:
        if cutname != args.cut:
            return

    print(f"Checking 2D content for: {variable}...")

    # Actually load the histogram into memory
    # This will internally convert ROOT histograms to hist
    h = hists.load(histkey)

    if "sys" in h.axes.name:
        h = replace_missing_systematics(h)

    # Group the histogram by datasets, using the dictionary defined
    # above from the config.
    # This will create a new coarse axis in the histogram, which we
    # call "process".
    h = group_histogram(h, old_axis="dataset", new_axis="process", mapping=groups)

    # Get a list of dense axes in the histogram
    dense_axes = [
        ax for ax in h.axes
        if not (isinstance(ax, hist.axis.StrCategory)
                or isinstance(ax, hist.axis.IntCategory))
    ]

    # Filter for 2D histograms only
    if len(dense_axes) != 2:
        # Silent return because most hists in the file are likely 1D
        return

    # Setup the x and y axes for the 2D plot.
    ax_x, ax_y = dense_axes[0], dense_axes[1]

    # Get a dictionary of all categories we want for this histogram
    # using the helper method defined above
    category_dict = get_category_dict(h, config)

    # Iterate over all combinations of categories defined earlier
    for category_label, category_definition in category_dict.items():

        # Make a subfolder for this cut and category
        outdir = args.outdir if args.outdir else os.path.dirname(args.histfile)
        cat_folder = os.path.join(outdir, cutname, category_label, "2D_plots")
        os.makedirs(cat_folder, exist_ok=True)

        # Build a output file name for the plot
        # The extension will be added in the plotting function,
        # in case we want to save multiple formats
        outfile = os.path.join(cat_folder, f"{variable}_{cutname}_{category_label}")

        # Slice the histogram according to the category definition
        h_cat = h
        for axname, axvals in category_definition.items():
            # Make sure every entry is a list, even if it is just one
            # Otherwise hist.integrate will give nonsense
            if not isinstance(axvals, list):
                axvals = [axvals]
            h_cat = h_cat.integrate(axname, axvals)

        try:
            # Plot with the method defined above
            plot_2d(h_cat, config, ax_x, ax_y, outfile, cutname, args.ext)
        except Exception:
            print(f"Error for variable {variable}, cutname {cutname}, "
                  f"category {category_label}:")
            traceback.print_exc()


if args.jobs == 1:
    # Iterate over all histograms in our collection
    for histkey in tqdm.tqdm(hists.keys()):
        process_histogram(histkey)

else:
    with multiprocessing.Pool(args.jobs) as pool:
        list(tqdm.tqdm(pool.imap_unordered(
            process_histogram, hists.keys()), total=len(hists)))
