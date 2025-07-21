import os
from pathlib import Path
import plotly.graph_objs as go

project_name = "dacs_ml"

# Base project directory (default to ~/projects/sac_ml)
base_dir = Path(os.environ.get("DACSML_PROJECT_DIR", Path.home() / "projects" / project_name))

# Define subdirectories
data_dir = base_dir / "data"
figs_dir = base_dir / "figs"
code_dir = base_dir / "code"
paper_figs_dir = figs_dir / "for_paper"


# ──────────────────────────────────────────
# Random Forest Parameters Dictionary
# ──────────────────────────────────────────

rf_dict = {
    "bootstrap": True,
    "ccp_alpha": 0.0,
    "criterion": "squared_error",
    "max_depth": 8,
    "max_features": 0.4,
    "max_leaf_nodes": None,
    "max_samples": None,
    "min_impurity_decrease": 0.0,
    # "min_impurity_split": None,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "min_weight_fraction_leaf": 0.0,
    "n_estimators": 128,
    "n_jobs": -1,
    "oob_score": False,
    "random_state": 0,
    "verbose": 0,
    "warm_start": False,
}

# ──────────────────────────────────────────
# from data.py
# ──────────────────────────────────────────
N_SPLITS = 5

rf_parameter_dict = {
    "bootstrap": True,
    "ccp_alpha": 0.0,
    "criterion": "squared_error",
    "max_depth": 8,
    "max_features": 0.4,
    "max_leaf_nodes": None,
    "max_samples": None,
    "min_impurity_decrease": 0.0,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "min_weight_fraction_leaf": 0.0,
    "n_estimators": 128,
    "n_jobs": -1,
    "oob_score": False,
    "random_state": 0,
    "verbose": 0,
    "warm_start": False,
}


# ──────────────────────────────────────────
# Regression Parity Plot Layout
# ──────────────────────────────────────────
regr_layout = go.Layout(
    width=597,
    height=597,
    font=dict(family="Arial", color="black"),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
    ),
    hoverlabel={"namelength": -1},
    paper_bgcolor="white",
    plot_bgcolor="white",
    legend=dict(
        xanchor="right",
        x=1,
        yanchor="bottom",
        y=0,
        bgcolor="rgba(0,0,0,0.1)",
        font_size=26,
        tracegroupgap=2,
    ),
    xaxis=dict(
        title_font_size=30,
        showline=True,
        linewidth=3,
        linecolor="black",
        mirror=True,
        showgrid=False,
        zeroline=False,
        gridcolor="rgba(0,0,0,0.3)",
        ticks="outside",
        tickfont_size=26,
        tickformat="d",
        tick0=200,
        dtick=200,
        tickwidth=3,
        ticklen=6,
    ),
    yaxis=dict(
        title_font_size=30,
        showline=True,
        linewidth=3,
        linecolor="black",
        mirror=True,
        showgrid=False,
        zeroline=False,
        gridcolor="rgba(0,0,0,0.3)",
        ticks="outside",
        tickfont_size=26,
        tickformat="d",
        tick0=200,
        dtick=200,
        tickwidth=3,
        ticklen=6,
    ),
)

# ──────────────────────────────────────────
# Energy layout
# ──────────────────────────────────────────

energy_layout = go.Layout(
    # Update global layout
    width=600,
    height=600,
    font=dict(family="Arial", color="black", size=26),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
    ),
    hoverlabel={"namelength": -1},
    # title=dict(text=plot_title, x=0.5, ),
    paper_bgcolor="white",
    plot_bgcolor="white",
    legend=dict(
        xanchor="right",
        x=1,
        yanchor="bottom",
        y=0,
        bgcolor="rgba(0,0,0,0.1)",  # bordercolor='rgba(0,0,0,0.4)',
        font_size=26,
        tracegroupgap=2,
    ),
    xaxis=dict(
        title="E<sub>DFT</sub> / eV",
        title_font_size=30,
        showline=True,
        linewidth=3,
        linecolor="black",
        mirror=True,
        showgrid=False,
        zeroline=False,
        ticks="outside",
        tickfont_size=26,
        tickformat=".1f",
        tickwidth=3,
        ticklen=6,
    ),
    yaxis=dict(
        title="E<sub>pred</sub> / eV",
        title_font_size=30,
        showline=True,
        linewidth=3,
        linecolor="black",
        mirror=True,
        showgrid=False,
        zeroline=False,
        ticks="outside",
        tickfont_size=26,
        tickformat=".1f",
        tickwidth=3,
        ticklen=6,
    ),
)

# ──────────────────────────────────────────
# Cross validation parameters
# ──────────────────────────────────────────

cv_kfold = {"cv_type": "kfold", "cv_spec": 5}
cv_logocv_metal = {"cv_type": "logocv", "cv_spec": "metal"}
cv_logocv_cavity = {
    "cv_type": "logocv",
    "cv_spec": "cavity",
}

# ──────────────────────────────────────────
# Various Color Palettes
# ──────────────────────────────────────────

default_colors = [
    "#1f77b4",  # muted blue
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#d62728",  # brick red
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # middle gray
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
] * 10

# Plotly Color Palette
rgba_colors = [
    "rgba(99,110,250,1)",
    "rgba(239,85,59,1)",
    "rgba(0,204, 150,1)",
    "rgba(171,99,250,1)",
    "rgba(255,161,90,1)",
]


set1_colors = [
    "rgb(55,126,184,1)",
    "rgb(77,175,74,1)",
    "rgb(152,78,163,1)",
    "rgb(255,127,0,1)",
]

hetatom_colors = [
    "rgba(169, 169, 169, 1)",  # grey C
    "rgba(42, 81, 104, 1)",  # blue N
    "rgba(162, 59, 62, 1)",  # red P
    "rgba(164, 117, 60, 1)",  # orange S
]

css_colors2 = [
    "#FD3216",
    "#00FE35",
    "#6A76FC",
    "#FED4C4",
    "#FE00CE",
    "#0DF9FF",
    "#F6F926",
    "#FF9616",
    "#479B55",
    "#EEA6FB",
    "#DC587D",
    "#D626FF",
    "#6E899C",
    "#00B5F7",
] * 10

default_symbols = [
    "circle",
    "square",
    "diamond",
    "triangle-up",
    "pentagon",
    "hexagram",
    "star",
    "hourglass",
    "bowtie",
    "cross",
    "x",
] * 10

none_symbols = [
    "circle",
    "circle",
    "circle",
    "circle",
    "circle",
    "circle",
    "circle",
    "circle",
    "circle",
    "circle",
    "circle",
] * 14


# Example color palette for 14 discrete categories
color_palette = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # yellow-green
    "#17becf",  # cyan
    "#aec7e8",  # light blue
    "#ffbb78",  # light orange
    "#98df8a",  # light green
    "#ff9896",  # light red
]

# ──────────────────────────────────────────
# Color setup for HPT figures
# ──────────────────────────────────────────

color_setup = {
    "train_color": "blue",      # Color for training data
    "test_color": "green",      # Color for testing data
    "line_color": "red",        # Color for regression line
    "scatter_color": "orange",  # Color for scatter plot
    "highlight_color": "black"  # Color for highlighted points
}

color_list = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # yellow-green
    "#17becf",  # cyan
    "#aec7e8",  # light blue
    "#ffbb78",  # light orange
    "#98df8a",  # light green
    "#ff9896",  # light red
]


# ─────────────────────────────────────────────────────────────────────
# Function for converting color codes from RGB to Hex 
# ─────────────────────────────────────────────────────────────────────

def rgb_to_hex(rgb):
    """
    Convert RGB color values to Hexadecimal format.

    Parameters:
    ----------
    rgb : tuple of int
        A tuple containing RGB values, each in the range 0-255.

    Returns:
    -------
    str
        The color in Hexadecimal format.
    """
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# ─────────────────────────────────────────────────────────────────────
# Color Definitions in Hex code
# ─────────────────────────────────────────────────────────────────────

# Updated metal_colors dictionary with hexadecimal color codes
metal_colors = {
    "Ru": rgb_to_hex((255, 0, 0)),  # Red
    "Rh": rgb_to_hex((0, 255, 0)),  # Green
    "Co": rgb_to_hex((0, 0, 255)),  # Blue
    "Fe": rgb_to_hex((255, 255, 0)),  # Yellow
    "Zn": rgb_to_hex((255, 0, 255)),  # Magenta
    "Ni": rgb_to_hex((0, 255, 255)),  # Cyan
    "Os": rgb_to_hex((128, 128, 0)),  # Olive
    "Cd": rgb_to_hex((128, 0, 128)),  # Purple
    "Pd": rgb_to_hex((0, 128, 128)),  # Teal
    "Cu": rgb_to_hex((128, 0, 0)),  # Maroon
    "Ag": rgb_to_hex((0, 128, 0)),  # Green (light)
    "Ir": rgb_to_hex((0, 0, 128)),  # Navy
    "Au": rgb_to_hex((255, 165, 0)),  # Orange
    "Pt": rgb_to_hex((128, 128, 128)),  # Gray
}

# ─────────────────────────────────────────────────────────────────────
# Cavity Dictionaries
# ─────────────────────────────────────────────────────────────────────

cavities = {
    "N_din4_x2_c5_b",
    "N_din6_s_c2_01",
    "N_din6_s_c5",
    "N_din6_as_c3_346",
    "N_din4_x2_c3_c_v2",
    "N_din6_as_c2_35",
    "N_din4_x2_c1_a_v2",
    "N_din4_x2_c4_f",
    "N_din6_as_c5_12345",
    "N_din6_as_c3_125",
    "N_din6_as_c3_253",
    "N_din4_x2_c4_d_v2",
    "N_din4_x2_c3_b_v2",
    "N_din6_s_c2_15",
    "N_din4_x2_c3_b",
    "N_din6_as_c3_165",
    "N_din4_x2_c5_b_v2",
    "N_din6_as_c3_123",
    "N_din4_x2_c2_g",
    "N_din6_as_c2_15",
    "N_din6_s_c2_12",
    "N_din4_x2_c2_g_v2",
    "N_din6_as_c2_25",
    "N_din6_s_c3_012",
    "N_din6_s_c4_0123",
    "N_din6_as_c2_23",
    "N_din6_as_c2_16",
    "N_din6_as_c2_34",
    "N_din4_x2_c4_d",
    "N_din6_as_c3_235",
    "N_din6_s_c0",
    "N_din4_x2_c4_c_v2",
    "N_din4_x2_c2_b",
    "N_din4_x2_c2_f",
    "N_din4_x2_c2_c",
    "N_din4_x2_c2_a_v2",
    "N_din6_s_c4_0134",
    "N_din4_x2_c4_a",
    "N_din6_s_c3_024",
    "N_din4_x2_c1_a",
    "C_din6_as_c6",
    "N_din6_s_c1",
    "N_din4_x2_c2_a",
    "N_din4_x2_c1_b",
    "N_din4_x2_c2_e_v2",
    "N_din6_as_c3_254",
    "N_din4_x2_c3_d",
    "N_din4_x2_c4_a_v2",
    "N_din6_as_c1_3",
    "N_din4_x2_c3_c",
    "N_din6_as_c3_124",
    "N_din6_s_c3_013",
    "N_din6_as_c2_13",
    "N_din4_x2_c2_d",
    "C_din6_s_c6",
    "N_din6_as_c4_6124",
    "N_din6_as_c5_12346",
    "N_din6_as_c3_216",
    "N_din6_as_c1_2",
    "N_din4_x2_c2_b_v2",
    "N_din6_as_c4_2346",
    "N_din4_x2_c4_f_v2",
    "N_din6_as_c2_36",
    "N_din6_as_c2_12",
    "N_din4_x2_c4_c",
    "N_din6_as_c2_26",
    "N_din6_as_c0",
    "N_din6_as_c4_6125",
    "N_din6_as_c3_163",
    "C_din4_x2_c6",
    "N_din6_as_c2_24",
    "N_din6_s_c2_02",
    "N_din6_as_c4_2456",
    "N_din6_s_c4_0124",
    "N_din4_x2_c3_d_v2",
    "N_din6_as_c3_236",
    "N_din6_as_c1_1",
    "N_din4_x2_c4_e",
    "N_din4_x2_c5_a",
    "N_din6_as_c4_3216",
    "N_din6_as_c4_3456",
    "N_din6_as_c3_164",
    "N_din4_x2_c0",
    "N_din6_as_c2_14",
    "N_din4_x2_c3_a",
    "N_din4_x2_c4_b",
    "N_din4_x2_c2_e",
}

# Define colors
color_din4_x2 = "red"
color_din6_s = "blue"
color_din6_as = "green"

# Initialize the dictionary
cavity_colors = {}

# Assign colors based on the presence of specific substrings
for cavity in cavities:
    if "din4_x2" in cavity:
        cavity_colors[cavity] = color_din4_x2
    elif "din6_s" in cavity:
        cavity_colors[cavity] = color_din6_s
    elif "din6_as" in cavity:
        cavity_colors[cavity] = color_din6_as


cavities_sacs = {
    "din3",
    "din3_c1",
    "din3_c2",
    "din3_c3",
    "din4",
    "din4_c1",
    "din4_c2_d1",
    "din4_c2_d2",
    "din4_c2_s1",
    "din4_c2_s2",
    "din4_c3",
    "din4_c4",
    "rol1_din2",
    "rol1_din2_c1_a",
    "rol1_din2_c1_b",
    "rol1_din2_c2_s1",
    "rol1_din2_c2_s2",
    "rol1_din2_c3",
    "rol2_din2",
    "rol2_din2_a",
    "rol2_din2_b",
    "rol2_din2_c2_d1",
    "rol2_din2_c2_s1",
    "rol2_din2_c2_s2",
    "rol2_din2_c2_s3",
    "rol2_din2_c3_a",
    "rol2_din2_c3_b",
    "rol2_din2_c4",
    "rol3",
    "rol3_c1",
    "rol3_c2",
    "rol3_c3",
}


# Define colors
color_din4 = "magenta"
color_rol2_din2 = "cyan"
color_rol3 = "purple"
color_rol1_din2 = "orange"
color_din3 = "yellow"


# Initialize the dictionary
cavity_colors_sacs = {}

# Assign colors based on the presence of specific substrings
for cavity in cavities_sacs:
    if "din4" in cavity:
        cavity_colors_sacs[cavity] = color_din4
    elif "rol2_din2" in cavity:
        cavity_colors_sacs[cavity] = color_rol2_din2
    elif "rol3" in cavity:
        cavity_colors_sacs[cavity] = color_rol3
    elif "rol1_din2" in cavity:
        cavity_colors_sacs[cavity] = color_rol1_din2
    elif "din3" in cavity:
        cavity_colors_sacs[cavity] = color_din3

# ─────────────────────────────────────────────────────────────────────
# Dual atom catalysts dictionary
# ─────────────────────────────────────────────────────────────────────

dacs_dict = {
    "C_din4_x2_c6": ["C_din4_c4", "C_din4_c4"],
    "N_din4_x2_c0": ["N_din4", "N_din4"],
    "N_din4_x2_c1_a": ["N_din4", "N_din4_c1"],
    "N_din4_x2_c1_b": ["N_din4_c1", "N_din4_c1"],
    "N_din4_x2_c2_a": ["N_din4", "N_din4_c2_s2"],
    "N_din4_x2_c2_b": ["N_din4_c1", "N_din4_c2_d1"],
    "N_din4_x2_c2_c": ["N_din4_c1", "N_din4_c1"],
    "N_din4_x2_c2_d": ["N_din4_c1", "N_din4_c1"],
    "N_din4_x2_c2_e": ["N_din4_c2_d1", "N_din4_c1"],
    "N_din4_x2_c2_f": ["N_din4_c2_s2", "N_din4_c2_s2"],
    "N_din4_x2_c2_g": ["N_din4_c2_s1", "N_din4_c1"],
    "N_din4_x2_c3_a": ["N_din4_c2_s1", "N_din4_c2_s1"],
    "N_din4_x2_c3_b": ["N_din4_c2_d2", "N_din4_c2_s1"],
    "N_din4_x2_c3_c": ["N_din4_c2_s2", "N_din4_c3"],
    "N_din4_x2_c3_d": ["N_din4_c1", "N_din4_c3"],
    "N_din4_x2_c4_a": ["N_din4_c3", "N_din4_c2_s1"],
    "N_din4_x2_c4_b": ["N_din4_c3", "N_din4_c3"],
    "N_din4_x2_c4_c": ["N_din4_c3", "N_din4_c2_d1"],
    "N_din4_x2_c4_d": ["N_din4_c3", "N_din4_c3"],
    "N_din4_x2_c4_e": ["N_din4_c2_s2", "N_din4_c2_s2"],
    "N_din4_x2_c4_f": ["C_din4_c4", "N_din4_c2_s2"],
    "N_din4_x2_c5_a": ["N_din4_c3", "N_din4_c3"],
    "N_din4_x2_c5_b": ["C_din4_c4", "N_din4_c3"],
    "C_din6_as_c6": ["C_din4_c4", "C_din4_c4"],
    "N_din6_as_c0": ["N_din4", "N_din4"],
    "N_din6_as_c1_1": ["N_din4", "N_din4_c1"],
    "N_din6_as_c1_2": ["N_din4_c1", "N_din4_c1"],
    "N_din6_as_c1_3": ["N_din4_c1", "N_din4"],
    "N_din6_as_c2_12": ["N_din4_c1", "N_din4_c2_s2"],
    "N_din6_as_c2_13": ["N_din4_c1", "N_din4_c1"],
    "N_din6_as_c2_14": ["N_din4_c1", "N_din4_c1"],
    "N_din6_as_c2_15": ["N_din4_c1", "N_din4_c2_d1"],
    "N_din6_as_c2_16": ["N_din4", "N_din4_c2_s1"],
    "N_din6_as_c2_23": ["N_din4_c2_s2", "N_din4_c1"],
    "N_din6_as_c2_24": ["N_din4_c2_d1", "N_din4_c1"],
    "N_din6_as_c2_25": ["N_din4_c2_s1", "N_din4_c2_s1"],
    "N_din6_as_c2_26": ["N_din4_c1", "N_din4_c2_d1"],
    "N_din6_as_c2_34": ["N_din4_c2_s1", "N_din4"],
    "N_din6_as_c2_35": ["N_din4_c2_d1", "N_din4_c2_s2"],
    "N_din6_as_c2_36": ["N_din4_c1", "N_din4_c1"],
    "N_din6_as_c3_123": ["N_din4_c2_s2", "N_din4_c2_s2"],
    "N_din6_as_c3_124": ["N_din4_c2_s1", "N_din4_c2_s2"],
    "N_din6_as_c3_125": ["N_din4_c2_s1", "N_din4_c3"],
    "N_din6_as_c3_163": ["N_din4_c1", "N_din4_c2_s1"],
    "N_din6_as_c3_164": ["N_din4_c1", "N_din4_c2_s1"],
    "N_din6_as_c3_165": ["N_din4_c1", "N_din4_c3"],
    "N_din6_as_c3_216": ["N_din4_c1", "N_din4_c3"],
    "N_din6_as_c3_235": ["N_din4_c3", "N_din4_c2_s1"],
    "N_din6_as_c3_236": ["N_din4_c2_s2", "N_din4_c2_d1"],
    "N_din6_as_c3_253": ["N_din4_c3", "N_din4_c2_s1"],
    "N_din6_as_c3_254": ["N_din4_c3", "N_din4_c2_s1"],
    "N_din6_as_c3_346": ["N_din4_c2_s1", "N_din4_c1"],
    "N_din6_as_c4_2346": ["N_din4_c3", "N_din4_c2_d2"],
    "N_din6_as_c4_2456": ["N_din4_c2_s1", "N_din4_c2_s1"],
    "N_din6_as_c4_3216": ["N_din4_c2_s2", "N_din4_c3"],
    "N_din6_as_c4_3456": ["N_din4_c3", "N_din4_c2_s2"],
    "N_din6_as_c4_6124": ["N_din4_c2_d1", "N_din4_c3"],
    "N_din6_as_c4_6125": ["N_din4_c2_s1", "C_din4_c4"],
    "N_din6_as_c5_12345": ["C_din4_c4", "N_din4_c3"],
    "N_din6_as_c5_12346": ["C_din4_c4", "N_din4_c3"],
    "C_din6_s_c6": ["C_din4_c4", "C_din4_c4"],
    "N_din6_s_c0": ["N_din4", "N_din4"],
    "N_din6_s_c1": ["N_din4_c1", "N_din4"],
    "N_din6_s_c2_01": ["N_din4", "N_din4_c2_s1"],
    "N_din6_s_c2_02": ["N_din4_c1", "N_din4_c2_d1"],
    "N_din6_s_c2_12": ["N_din4_c1", "N_din4_c1"],
    "N_din6_s_c2_15": ["N_din4_c1", "N_din4_c2_s1"],
    "N_din6_s_c3_012": ["N_din4_c1", "N_din4_c3"],
    "N_din6_s_c3_013": ["N_din4_c1", "N_din4_c2_s2"],
    "N_din6_s_c3_024": ["N_din4_c2_d1", "N_din4_c2_d1"],
    "N_din6_s_c4_0123": ["N_din4_c2_s1", "N_din4_c3"],
    "N_din6_s_c4_0124": ["N_din4_c2_d1", "N_din4_c3"],
    "N_din6_s_c4_0134": ["N_din4_c2_s1", "N_din4_c2_s1"],
    "N_din6_s_c5": ["N_din4_c3", "N_din4_c3"],
}
