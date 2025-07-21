# ─────────────────────────────────────────────────────────────
#  Standard Library
# ─────────────────────────────────────────────────────────────
import os
import copy
import logging
from typing import Hashable, List

# ─────────────────────────────────────────────────────────────
#  Scientific Libraries
# ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────
#  Plotly Visualization
# ─────────────────────────────────────────────────────────────
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────
#  External Libraries
# ─────────────────────────────────────────────────────────────
from ase import Atoms
from regex import D, W  # <- Confirm usage or remove if unused

# ─────────────────────────────────────────────────────────────
#  Internal Project Modules
# ─────────────────────────────────────────────────────────────
from settings import *



# ─────────────────────────────────────────────────────────────────────
# Function to save an image to png, scv, html
# ─────────────────────────────────────────────────────────────────────

def plotly_to_image(
    plotly_fig: go.Figure,
    path_elements: List[str],
    figure_name: str,
    save_types: list = ["png", "svg", "html"],
    paper: bool = False,
):
    """This function takes a plotly figure object and writes it as svg, png, and html files to disk. In particular, it takes care of removing the `non-scaling-stroke` vector-effect in the svg, and, if need be, creates an svg without annotations for postprocessing in Inkscape.

    Args:
        plotly_fig (go.Figure): Plotly.go Figure object to be saved.
        path_elements (list[str]): List of path elements for the save location of the Figure.
        figure_name (str): Base filename for the saved Figure.
        save_types (list, optional): Which formats shall be saved. Defaults to ["png", "svg", "html"].
        paper (bool, optional): If an additional version of the svg without ticks or annotations should be saved. Defaults to False.

    Returns:
        None
    """

    # ! This could also be done outside...
    main_path = os.path.join(*path_elements)

    for save_type in save_types:
        if save_type == "svg":
            # ! All the extra work for svgs to remove vector-effect, and create figures without annotations for the paper.
            svg_filename = os.path.join(main_path, "{}.svg".format(figure_name))
            # str.replace() returns a new string, old string is not changed.
            temp_filename = svg_filename.replace(".svg", "_temp.svg")

            plotly_fig.write_image(temp_filename, engine="kaleido")
            with open(temp_filename, "rt") as fin:
                with open(svg_filename, "wt") as fout:
                    for line in fin:
                        fout.write(
                            line.replace(
                                "vector-effect: non-scaling-stroke",
                                "vector-effect: none",
                            )
                        )
            os.remove(temp_filename)

            if paper is True:
                svg_filename = svg_filename.replace(".svg", "_paper.svg")
                paper_plotly_fig = copy.deepcopy(plotly_fig)
                for anno in paper_plotly_fig["layout"]["annotations"]:
                    anno["text"] = ""
                paper_layout = go.Layout(
                    xaxis=dict(ticks="", showticklabels=False, showgrid=False),
                    yaxis=dict(ticks="", showticklabels=False, showgrid=False),
                    title_text="",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                _ = paper_plotly_fig.update_layout(paper_layout)

                paper_plotly_fig.write_image(temp_filename, engine="kaleido")

                with open(temp_filename, "rt") as fin:
                    with open(svg_filename, "wt") as fout:
                        for line in fin:
                            fout.write(
                                line.replace(
                                    "vector-effect: non-scaling-stroke",
                                    "vector-effect: none",
                                )
                            )
                os.remove(temp_filename)

        # Here, we still create it from path_elements, as defining main_path does not provide much gain/advantage.
        elif save_type == "html":
            plotly_fig.write_html(
                os.path.join(main_path, "{}.{}".format(figure_name, save_type))
            )

        else:  # png or pdf
            plotly_fig.write_image(
                os.path.join(main_path, "{}.{}".format(figure_name, save_type)),
                engine="kaleido",
            )

    return None

# ─────────────────────────────────────────────────────────────────────
# Map categorical values to values
# ─────────────────────────────────────────────────────────────────────

def create_mapping_dict(
    df_in: pd.DataFrame, column: list = None, value_list: list = None
):
    if column is None and value_list is None:
        return {}
    elif column is not None and value_list is not None:
        # df_out = df_out.dropna(subset=[column])
        df_out = df_in.copy(deep=True)
        column_values = sorted(list(set(list(df_out[column].values))))
        mapping_dict = {
            k: v
            for k, v in list(zip(column_values, value_list[0 : len(column_values)]))
        }

        return mapping_dict
    else:
        raise ValueError("Either both or none of the parameters must be set.")

# ─────────────────────────────────────────────────────────────────────
# Generate Error Summary Text
# ─────────────────────────────────────────────────────────────────────

def create_error_text(error_dict, which_error):
    if which_error != "all":
        if which_error == "mean":
            error_list = [
                np.mean(error_dict["rsquared_tests"]),
                np.std(error_dict["rsquared_tests"]),
                np.mean(error_dict["rmse_tests"]),
                np.std(error_dict["rmse_tests"]),
                np.mean(error_dict["mae_tests"]),
                np.std(error_dict["mae_tests"]),
            ]
        elif which_error == "best":
            error_list = [
                np.max(error_dict["rsquared_tests"]),
                np.std(error_dict["rsquared_tests"]),
                np.min(error_dict["rmse_tests"]),
                np.std(error_dict["rmse_tests"]),
                np.min(error_dict["mae_tests"]),
                np.std(error_dict["mae_tests"]),
            ]
        elif which_error == "full_mean":
            error_list = [
                np.mean(error_dict["rsquared_fulls"]),
                np.std(error_dict["rsquared_fulls"]),
                np.mean(error_dict["rmse_fulls"]),
                np.std(error_dict["rmse_fulls"]),
                np.mean(error_dict["mae_fulls"]),
                np.std(error_dict["mae_fulls"]),
            ]
        elif which_error == "full_best":
            error_list = [
                np.max(error_dict["rsquared_fulls"]),
                np.std(error_dict["rsquared_fulls"]),
                np.min(error_dict["rmse_fulls"]),
                np.std(error_dict["rmse_fulls"]),
                np.min(error_dict["mae_fulls"]),
                np.std(error_dict["mae_fulls"]),
            ]
        else:
            raise KeyError("Invalid error specification used.")

        error_text = "R<sup>2</sup> = {:.3f} &#177; {:.3f}<br>RMSE = {:.3f} &#177; {:.3f}<br>MAE = {:.3f} &#177; {:.3f}".format(
            *error_list
        )
        # ! Change decimal to 3
    else:
        error_list_train = [
            np.mean(error_dict["rsquared_trains"]),
            np.std(error_dict["rsquared_trains"]),
            np.mean(error_dict["rmse_trains"]),
            np.std(error_dict["rmse_trains"]),
            np.mean(error_dict["mae_trains"]),
            np.std(error_dict["mae_trains"]),
        ]
        error_text = "Train:<br>R<sup>2</sup> = {:.2f} &#177; {:.2f}<br>RMSE = {:.2f} &#177; {:.2f}<br>MAE = {:.2f} &#177; {:.2f}".format(
            *error_list_train
        )
        error_list_test = [
            np.mean(error_dict["rsquared_tests"]),
            np.std(error_dict["rsquared_tests"]),
            np.mean(error_dict["rmse_tests"]),
            np.std(error_dict["rmse_tests"]),
            np.mean(error_dict["mae_tests"]),
            np.std(error_dict["mae_tests"]),
        ]

        error_text += "<br>Test:<br>R<sup>2</sup> = {:.2f} &#177; {:.2f}<br>RMSE = {:.2f} &#177; {:.2f}<br>MAE = {:.2f} &#177; {:.2f}".format(
            *error_list_test
        )
    return error_text


# ─────────────────────────────────────────────────────────────────────
# Lowest Cardinality Function
# ─────────────────────────────────────────────────────────────────────

def get_df_cardinality_col(df_in, which: str = "lowest"):
    nunique_series = df_in.select_dtypes(include=["number"]).nunique()
    cardinality_column = nunique_series[nunique_series > 0].idxmin()
    return cardinality_column

# ─────────────────────────────────────────────────────────────────────
# Train/Test Scatter Traces to Regression Plot 
# ─────────────────────────────────────────────────────────────────────

def _add_train_test_trace(
    df_in,
    regr_fig,
    color_column,
    symbol_column,
    text_column,
    symbol_mapping,
    color_mapping,
    cv_id_pred_column,
    which_trace,
    **kwargs,
):
    DEFAULT_COLOR_DICT = {"train": "black", "test": "red"}
    DEFAULT_SYMBOL_DICT = {"train": "circle", "test": "cross"}

    if color_column is None:
        color_column = df_in.nunique().idxmin()

    color_column_values = df_in[color_column].unique()

    for color_column_value in color_column_values:
        column_train_df = df_in.loc[df_in[color_column] == color_column_value]
        if text_column is not None:
            plot_text = column_train_df[text_column]
        else:
            plot_text = [""] * column_train_df.shape[0]

        symbols = [
            symbol_mapping.get(_, DEFAULT_SYMBOL_DICT[which_trace])
            for _ in column_train_df[symbol_column].values
        ]

        colors = color_mapping.get(color_column_value, DEFAULT_COLOR_DICT[which_trace])

        _ = regr_fig.add_trace(
            go.Scatter(
                x=column_train_df["y"],
                y=column_train_df[cv_id_pred_column],
                mode="markers",
                marker=dict(
                    size=12,
                    symbol=symbols,
                    opacity=1,
                    color=colors,
                    line=dict(
                        color="black",
                        # color=color_mapping.get(column_value, "black"),  # Rigid black color for the  perimeter
                        width=2,
                    ),  # Adjust the width of the  perimeter
                ),
                hoverinfo="text+x+y",
                name="{}".format(color_column_value),
                # name=kwargs.get("legendgroup", f"{column_value}"),
                text=plot_text,
                legendgroup="{}".format(color_column_value),
                # legendgroup=kwargs.get("legendgroup", f"{column_value}"),
                showlegend=kwargs.get("show_train_legend", True),
            ),
        )
    return regr_fig


# ─────────────────────────────────────────────────────────────────────
# Cross-Validation Regression Results Plot - General
# ─────────────────────────────────────────────────────────────────────

def plot_regr(
    regr_dict,
    color_column=None,
    color_list=None,
    symbol_column=None,
    symbol_list=None,
    show_train=True,
    show_test=True,
    set_range=None,
    which_error="mean",
    regr_layout=None,
    axes_layout={},
    text_column=None,
    **kwargs,
):
    """
    Generate scatter plots for regression results.

    Args:
        regr_dict (dict): A dictionary containing regression results and data.
        color_column (str): The name of the column in the dataset to use for coloring the data points.
        show_train (bool, optional): Whether to show the training data points. Defaults to True.
        show_test (bool, optional): Whether to show the test data points. Defaults to True.
        set_range (tuple, optional): The range of values to be displayed on the x and y axes. If None, the range is determined automatically. Defaults to None.
        which_error (str, optional): The type of error to display in the annotation. Possible values: "mean", "best", "full_mean", "full_best", "all". Defaults to "mean".
        color_mapping (dict, optional): A dictionary mapping column values to color names or hex values. If None, default colors are used. Defaults to None.
        regr_layout (dict, optional): Additional layout options for the plotly figure. Defaults to None.
        text_column (str, optional): The name of the column in the dataset to use for text labels on the data points. Defaults to None.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        list: A list of plotly figures representing the regression scatter plots.

    Raises:
        KeyError: If an invalid error specification is used for `which_error`.

    Notes:
        - The `regr_dict` parameter should contain the following keys: 'error_dict', 'df_in'.
        - The 'error_dict' should contain error metrics for train and test sets, e.g., 'rsquared_trains', 'rsquared_tests', 'rmse_trains', 'rmse_tests', etc.
        - The 'df_in' should contain the dataset used for regression, including the columns specified in `color_column` and `text_column` if applicable.
    """
    # function_arguments = locals()
    # print(type(kwargs))
    # print(type(function_arguments))
    # print(function_arguments.keys())

    # Include threshold for deviation somewhere
    # TODO: When doing LOOCV, the excluded metal (group) is not added to the legend.
    # TODO: Add possibility to provide specific colors. Had to remove that from the metal-color dict so that the code works general.
    # TODO: All plotly layout options using kwargs: `showticklabels`, `showlegend`, `label_column`

    # # Print the color names and their corresponding hex values
    # for color_name, hex_value in color_list.items():
    #     print(f"{color_name}: {hex_value}")
    error_dict = regr_dict["error_dict"]

    df_func = regr_dict["df_in"].copy(deep=True)

    # ! Sort df so that legend items appear ordered -> Check that this does not
    # mess up the ordering from input to pred values.

    color_mapping = create_mapping_dict(
        df_in=df_func, column=color_column, value_list=color_list
    )
    symbol_mapping = create_mapping_dict(
        df_in=df_func, column=symbol_column, value_list=symbol_list
    )

    regr_figs = []

    bool_columns = [col for col in df_func.columns if "train" in col]
    pred_columns = [col for col in df_func.columns if "pred" in col]

    for cv_id in range(len(error_dict["rmse_trains"])):
        cv_id_bool_column = bool_columns[int(cv_id)]
        cv_id_pred_column = pred_columns[int(cv_id)]

        split_bool_array = df_func[cv_id_bool_column].values

        df_train = df_func[split_bool_array].copy(deep=True)
        df_test = df_func[np.logical_not(split_bool_array)].copy(deep=True)

        # Instantiate figure
        regr_fig = go.Figure()

        # Add annotation with R^2 and RMSEs
        error_text = create_error_text(error_dict=error_dict, which_error=which_error)

        _ = regr_fig.add_annotation(
            xanchor="left",
            yanchor="top",
            xref="paper",
            yref="paper",
            x=0,
            y=1,
            align="left",
            text=error_text,
            font_size=26,
            font_family="Arial",
            showarrow=False,
            bgcolor="rgba(0,0,0,0.1)",
        )

        # Assign lowest-cardinality df column to color_column and symbol_column
        # if they are not set
        if color_column is None:
            color_column = get_df_cardinality_col(df_in=df_func)

        if symbol_column is None:
            symbol_column = get_df_cardinality_col(df_in=df_func)

        # ?Plot energy data points
        if show_train is True:
            regr_fig = _add_train_test_trace(
                df_in=df_train,
                regr_fig=regr_fig,
                color_column=color_column,
                symbol_column=symbol_column,
                symbol_mapping=symbol_mapping,
                color_mapping=color_mapping,
                text_column=text_column,
                cv_id_pred_column=cv_id_pred_column,
                which_trace="train",
                kwargs=kwargs,
            )

        if show_test is True:
            regr_fig = _add_train_test_trace(
                df_in=df_test,
                regr_fig=regr_fig,
                color_column=color_column,
                symbol_column=symbol_column,
                symbol_mapping=symbol_mapping,
                color_mapping=color_mapping,
                text_column=text_column,
                cv_id_pred_column=cv_id_pred_column,
                which_trace="test",
                kwargs=kwargs,
            )

        # todo: Move this out to separate function
        if set_range is None:
            all_values = (
                df_train["y"].tolist()
                + df_train[cv_id_pred_column].tolist()
                + df_test["y"].tolist()
                + df_test[cv_id_pred_column].tolist()
            )
            all_values = list(map(float, all_values))

            full_range = [min(all_values), max(all_values)]
            range_ext = (
                full_range[0] - 0.075 * np.ptp(full_range),
                full_range[1] + 0.075 * np.ptp(full_range),
            )
        else:
            range_ext = set_range

        # Add ideal fit line to plot
        _ = regr_fig.add_trace(
            go.Scatter(
                x=range_ext,
                y=range_ext,
                mode="lines",
                line=dict(color="black", width=2, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            ),
        )

        # Update global layout
        if regr_layout is not None:
            _ = regr_fig.update_layout(regr_layout)

        _axes_layout = go.Layout(
            {
                **axes_layout,
                **{"xaxis_range": range_ext, "yaxis_range": range_ext},
            }
        )

        _ = regr_fig.update_layout(_axes_layout)

        regr_figs.append(regr_fig)

    return regr_figs

# ─────────────────────────────────────────────────────────────────────
# Cross-Validation Regression Results Plot 
# ─────────────────────────────────────────────────────────────────────


def plot_regr_train_test(
    regr_dict,
    color_column,
    show_train=True,
    show_test=True,
    set_range=None,
    which_error="mean",
    color_mapping=None,
    regr_layout=None,
    text_column=None,
    *args,
    **kwargs,
):
    """
    Generate scatter plots for regression results.

    Args:
        regr_dict (dict): A dictionary containing regression results and data.
        color_column (str): The name of the column in the dataset to use for coloring the data points.
        show_train (bool, optional): Whether to show the training data points. Defaults to True.
        show_test (bool, optional): Whether to show the test data points. Defaults to True.
        set_range (tuple, optional): The range of values to be displayed on the x and y axes. If None, the range is determined automatically. Defaults to None.
        which_error (str, optional): The type of error to display in the annotation. Possible values: "mean", "best", "full_mean", "full_best", "all". Defaults to "mean".
        color_mapping (dict, optional): A dictionary mapping column values to color names or hex values. If None, default colors are used. Defaults to None.
        regr_layout (dict, optional): Additional layout options for the plotly figure. Defaults to None.
        text_column (str, optional): The name of the column in the dataset to use for text labels on the data points. Defaults to None.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        list: A list of plotly figures representing the regression scatter plots.

    Raises:
        KeyError: If an invalid error specification is used for `which_error`.

    Notes:
        - The `regr_dict` parameter should contain the following keys: 'error_dict', 'df_in'.
        - The 'error_dict' should contain error metrics for train and test sets, e.g., 'rsquared_trains', 'rsquared_tests', 'rmse_trains', 'rmse_tests', etc.
        - The 'df_in' should contain the dataset used for regression, including the columns specified in `color_column` and `text_column` if applicable.
    """
    # function_arguments = locals()
    # print(type(kwargs))
    # print(type(function_arguments))
    # print(function_arguments.keys())

    # # Print the color names and their corresponding hex values
    # for color_name, hex_value in color_list.items():
    #     print(f"{color_name}: {hex_value}")
    error_dict = regr_dict["error_dict"]

    df_func = regr_dict["df_in"].copy(deep=True)

    # if color_setup is None:
    #     text_colum
    #     color_setup = {}

    # color_setup = {k:v for k,v in }
    # print(color_setup)

    # color_column = list(color_setup.keys())[0]
    # color_mapping = list(color_setup.values())[0]

    
    if color_column is not None:
        df_func = df_func.dropna(subset=[color_column])
        color_column_values = sorted(list(set(list(df_func[color_column].values))))
        color_mapping = {
            k: v
            for k, v in list(
                zip(color_column_values, color_list[0 : len(color_column_values)])
            )
        }

        df_func = df_func.sort_values(by=color_column)

    regr_figs = []

    bool_columns = [col for col in df_func.columns if "train" in col]
    pred_columns = [col for col in df_func.columns if "pred" in col]

    for cv_id in range(len(error_dict["rmse_trains"])):
        cv_id_bool_column = bool_columns[int(cv_id)]
        cv_id_pred_column = pred_columns[int(cv_id)]

        split_bool_array = df_func[cv_id_bool_column].values

        df_train = df_func[split_bool_array].copy(deep=True)
        df_test = df_func[np.logical_not(split_bool_array)].copy(deep=True)

        # Instantiate figure
        regr_fig = go.Figure()


        if which_error != "all":
            if which_error == "mean":
                error_list = [
                    np.mean(error_dict["rsquared_tests"]),
                    np.std(error_dict["rsquared_tests"]),
                    np.mean(error_dict["rmse_tests"]),
                    np.std(error_dict["rmse_tests"]),
                    np.mean(error_dict["mae_tests"]),
                    np.std(error_dict["mae_tests"]),
                ]
            elif which_error == "best":
                error_list = [
                    np.max(error_dict["rsquared_tests"]),
                    np.std(error_dict["rsquared_tests"]),
                    np.min(error_dict["rmse_tests"]),
                    np.std(error_dict["rmse_tests"]),
                    np.min(error_dict["mae_tests"]),
                    np.std(error_dict["mae_tests"]),
                ]
            elif which_error == "full_mean":
                error_list = [
                    np.mean(error_dict["rsquared_fulls"]),
                    np.std(error_dict["rsquared_fulls"]),
                    np.mean(error_dict["rmse_fulls"]),
                    np.std(error_dict["rmse_fulls"]),
                    np.mean(error_dict["mae_fulls"]),
                    np.std(error_dict["mae_fulls"]),
                ]
            elif which_error == "full_best":
                error_list = [
                    np.max(error_dict["rsquared_fulls"]),
                    np.std(error_dict["rsquared_fulls"]),
                    np.min(error_dict["rmse_fulls"]),
                    np.std(error_dict["rmse_fulls"]),
                    np.min(error_dict["mae_fulls"]),
                    np.std(error_dict["mae_fulls"]),
                ]
            else:
                raise KeyError("Invalid error specification used.")

            error_text = "R<sup>2</sup> = {:.4f} &#177; {:.4f}<br>RMSE = {:.4f} &#177; {:.4f}<br>MAE = {:.4f} &#177; {:.4f}".format(
                *error_list
            )
        # ! Change decimal to 3
        else:
            error_list_train = [
                np.mean(error_dict["rsquared_trains"]),
                np.std(error_dict["rsquared_trains"]),
                np.mean(error_dict["rmse_trains"]),
                np.std(error_dict["rmse_trains"]),
                np.mean(error_dict["mae_trains"]),
                np.std(error_dict["mae_trains"]),
            ]
            error_text = "Train:<br>R<sup>2</sup> = {:.4f} &#177; {:.4f}<br>RMSE = {:.4f} &#177; {:.4f}<br>MAE = {:.4f} &#177; {:.4f}".format(
                *error_list_train
            )
            error_list_test = [
                np.mean(error_dict["rsquared_tests"]),
                np.std(error_dict["rsquared_tests"]),
                np.mean(error_dict["rmse_tests"]),
                np.std(error_dict["rmse_tests"]),
                np.mean(error_dict["mae_tests"]),
                np.std(error_dict["mae_tests"]),
            ]

            error_text += "<br>Test:<br>R<sup>2</sup> = {:.4f} &#177; {:.4f}<br>RMSE = {:.4f} &#177; {:.4f}<br>MAE = {:.4f} &#177; {:.4f}".format(
                *error_list_test
            )

        _ = regr_fig.add_annotation(
            xanchor="left",
            yanchor="top",
            xref="paper",
            yref="paper",
            x=0,
            y=1,
            align="left",
            text=error_text,
            font_size=26,
            font_family="Arial",
            showarrow=False,
            bgcolor="rgba(0,0,0,0.1)",
        )

        # print("color_column", color_column)
        if show_train is True:
            for column_value in df_train[color_column].unique():
                column_train_df = df_train.loc[df_train[color_column] == column_value]
                if text_column is not None:
                    plot_text = column_train_df[text_column]
                else:
                    plot_text = [""] * column_train_df.shape[0]
                ###
                _ = regr_fig.add_trace(
                    go.Scatter(
                        x=column_train_df["y"],
                        y=column_train_df[cv_id_pred_column],
                        mode="markers",
                        marker=dict(
                            size=12,  # initial plots 12, abstract 18
                            symbol=0,
                            opacity=1,  # initially opacity to 1
                            color="rgba(136,86,167,1)",  # "rgba(0,156,156, 0.7)",  # Blue color with 50% transparency
                            # color=color_mapping.get(column_value, "black"),
                            line=dict(
                                color="black",  # Rigid red color for the circle perimeter
                                width=1,
                            ),  # Adjust the width of the circle perimeter
                        ),
                        hoverinfo="text+x+y",
                        name="{}".format(column_value),
                        text=plot_text,
                        legendgroup="{}".format(column_value),
                        showlegend=kwargs.get("show_train_legend", True),
                    ),
                )

        if show_test is True:
            for column_value in df_test[color_column].unique():
                column_test_df = df_test.loc[df_test[color_column] == column_value]
                if text_column is not None:
                    plot_text = column_test_df[text_column]
                else:
                    plot_text = "" * column_test_df.shape[0]
                ###
                _ = regr_fig.add_trace(
                    go.Scatter(
                        x=column_test_df["y"],
                        y=column_test_df[cv_id_pred_column],
                        mode="markers",
                        marker=dict(
                            size=12,  # initially 8 , for paper 12, abstract 18
                            symbol=0,  # initially 4 for cross
                            opacity=1,  # initially opacity to 1
                            color="rgba(28,144,153,1)",  # "rgba(156,0,0, 0.7)",  # change to red the test set
                            # color=color_mapping.get(column_value, "black"),
                            line=dict(
                                color="black",  # Rigid red color for the circle perimeter
                                width=1,
                            ),  # Adjust the width of the circle perimeter
                        ),
                        hoverinfo="text+x+y",
                        name="{}".format(column_value),
                        text=plot_text,
                        legendgroup="{}".format(column_value),
                        showlegend=kwargs.get("show_test_legend", False),
                    ),
                )

        if set_range is None:
            all_values = (
                df_train["y"].tolist()
                + df_train[cv_id_pred_column].tolist()
                + df_test["y"].tolist()
                + df_test[cv_id_pred_column].tolist()
            )
            all_values = list(map(float, all_values))

            full_range = [min(all_values), max(all_values)]
            range_ext = (
                full_range[0] - 0.075 * np.ptp(full_range),
                full_range[1] + 0.075 * np.ptp(full_range),
            )
        else:
            range_ext = set_range

        # Add ideal fit line to plot
        _ = regr_fig.add_trace(
            go.Scatter(
                x=range_ext,
                y=range_ext,
                mode="lines",
                line=dict(color="black", width=2, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            ),
        )

        # Update global layout

        if regr_layout is not None:
            _ = regr_fig.update_layout(regr_layout)

        axes_layout = go.Layout(
            xaxis=dict(range=range_ext), yaxis=dict(range=range_ext)
        )
        _ = regr_fig.update_layout(axes_layout)

        regr_figs.append(regr_fig)

    return regr_figs

# ─────────────────────────────────────────────────────────────────────
# Plot Regression Error Metrics vs. Parameter (HPT) 
# ─────────────────────────────────────────────────────────────────────

def plot_errors(
    error_dict, x_values, plot_measures, annot_text, x_title, showlegend=True
):
    """
    Plot error measures (RMSE, MAE, R^2) for a machine learning model, obtained by varying a parameter.

    Args:
        error_dict (dict): Dictionary containing the error measures.
        x_values (list): List of x-axis values for the parameter being varied.
        plot_measures (list): List of error measures to plot (e.g., ['rmse_train', 'rmse_test', 'mae_train', 'rsquared_test']).
        annot_text (list): List of annotation texts corresponding to each data point on the plot.
        x_title (str): Title for the x-axis.
        showlegend (bool, optional): Whether to show the legend. Defaults to True.

    Returns:
        error_fig (plotly.graph_objects.Figure): Plotly figure object containing the error plot.
    """
    annot_texts = ["<br>".join(sorted(annot_text)) for annot_text in annot_text]

    error_fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces for error measures.
    for iplot_measure, plot_measure in enumerate(plot_measures):
        if "rmse" in plot_measure:
            line_color = "blue"
            line_name = "RMSE ({})".format(plot_measure.split("_")[1][:-1])
            y_axis = "y1"
        elif "mae" in plot_measure:
            line_color = "green"
            line_name = "MAE ({})".format(plot_measure.split("_")[1][:-1])
            y_axis = "y1"
        elif "rsquared" in plot_measure:
            line_color = "red"
            line_name = "R<sup>2</sup> ({})".format(plot_measure.split("_")[1][:-1])
            y_axis = "y2"
        if "test" in plot_measure:
            dash = "dash"
        else:
            dash = None

        _ = error_fig.add_trace(
            go.Scatter(
                x=np.array(x_values).astype(float).round(2),  #!
                # y=error_dict[
                #     plot_measure
                # ],
                y=np.array(error_dict[plot_measure]).astype(float).round(2),
                # TODO: Round numbers shown on hover (but not for plotting).
                mode="lines",
                name=line_name,
                line=dict(
                    color=line_color,
                    width=3,
                    dash=dash,
                ),
                showlegend=showlegend,
                hoverinfo="x+y",
                yaxis=y_axis,
            ),
            secondary_y="2" in y_axis,
        )

        if min(x_values) < 0.05:
            x_range = [0, max(x_values)]
        else:
            x_range = [min(x_values), max(x_values)]

        # Add invisible trace to allow for dynamic annotation of descriptors
        if iplot_measure == len(plot_measures) - 1:
            _ = error_fig.add_trace(
                go.Scatter(
                    # x=list(range(1, error_array.shape[0])),
                    x=np.array(x_values).astype(float).round(2),
                    y=error_dict[plot_measure],
                    mode="lines",
                    name="RMSE (train)",
                    line=dict(color="blue", width=0.001),
                    showlegend=False,
                    text=annot_texts,
                    hoverinfo="text",
                ),
            )

    error_layout = go.Layout(
        width=809,
        height=500,
        font=dict(family="Arial", color="black"),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            xanchor="center",
            x=0.5,
            yanchor="top",
            y=1,
            font_size=26,
            bgcolor="rgba(0,0,0,0.1)",
        ),
        hoverlabel={"namelength": -1},
        hovermode="x unified",
        xaxis=dict(
            title=x_title,
            title_font_size=30,
            range=x_range,
            showline=True,
            linewidth=3,
            linecolor="black",
            mirror=True,
            showgrid=False,
            zeroline=False,
            gridcolor="rgba(0,0,0,0.3)",
            ticks="outside",
            tickfont_size=26,
            tickwidth=3,
            ticklen=6,
        ),
        yaxis=dict(
            title="RMSE / eV",
            title_font_size=30,
            title_font_color="blue",  # range=[0, rmse_max*1.25],
            showline=True,
            linewidth=3,
            linecolor="black",
            color="blue",  # mirror=True,
            showgrid=False,
            zeroline=False,
            gridcolor="rgba(0,0,0,0.3)",  # tick0=0, dtick=0.1,
            ticks="outside",
            tickfont_size=26,
            tickwidth=3,
            ticklen=6,
        ),
        yaxis2=dict(
            title="R<sup>2</sup>",
            title_font_size=30,
            title_font_color="red",  # range=[0, 1],
            showline=True,
            linewidth=3,
            linecolor="black",
            color="red",  # mirror=True,
            showgrid=False,
            zeroline=False,
            gridcolor="rgba(0,0,0,0.3)",  # tick0=0, dtick=0.2,
            ticks="outside",
            tickfont_size=26,
            tickwidth=3,
            ticklen=6,
        ),
    )

    _ = error_fig.update_layout(error_layout)

    return error_fig



# ─────────────────────────────
# Feature Importance Plot
# ─────────────────────────────

def plot_feature_importance(
    feature_importance: dict,
    feature_name_mapping: dict = None,
    color_map: dict = None,
    figsize=(4, 5),
    title="Feature Importance",
    xlabel="Importance",
    ylabel="Features",
    save_dir: str = None,
    filename: str = "feature_importance",
    save_formats: list = ["png", "svg", "pdf"],
    dpi: int = 300,
    show: bool = True
):
    """
    Plots a horizontal bar chart of feature importances and optionally saves it.

    Parameters:
    - feature_importance: dict
    - feature_name_mapping: dict, optional
    - color_map: dict, optional
    - figsize: tuple, optional
    - title, xlabel, ylabel: str, optional
    - save_dir: str, optional
        Directory where the plot will be saved. If None, does not save.
    - filename: str
        Name of the saved file (without extension).
    - save_formats: list
        List of formats to save (e.g., ["png", "svg", "pdf"]).
    - dpi: int
        Resolution for raster formats like PNG.
    - show: bool
        Whether to display the plot inline (default True).
    """

    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1])
    features, importance_values = zip(*sorted_features)

    # Apply LaTeX-style feature names
    if feature_name_mapping:
        feature_names = [feature_name_mapping.get(f, f) for f in features]
    else:
        feature_names = features

    # Set up bar colors
    if color_map:
        bar_colors = [color_map.get(f, 'gray') for f in features]
    else:
        bar_colors = ['gray'] * len(features)

    # Convert RGB strings to matplotlib color tuples
    bar_colors = [
        tuple(int(c) / 255.0 for c in color[4:-1].split(',')) if isinstance(color, str) and color.startswith('rgb') else color
        for color in bar_colors
    ]

    # Plotting
    plt.figure(figsize=figsize)
    plt.grid(axis='x', linestyle='-', alpha=1, zorder=0)
    y_pos = np.arange(len(features))
    plt.barh(y_pos, importance_values, color=bar_colors, edgecolor=None, zorder=3)
    plt.yticks(y_pos, feature_names, fontsize=8, zorder=3)
    plt.xlabel(xlabel, zorder=3)
    plt.ylabel(ylabel, zorder=3)
    plt.title(title, zorder=3)
    plt.tight_layout()

    # Save plots if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for fmt in save_formats:
            path = os.path.join(save_dir, f"{filename}.{fmt}")
            plt.savefig(path, format=fmt, dpi=dpi if fmt == "png" else None)

    # Show the plot (optional)
    if show:
        plt.show()
    else:
        plt.close()


# ─────────────────────────────
# Pearson Correlation Matrix Plot
# ─────────────────────────────


def plot_pearson_correlation(
    df: pd.DataFrame,
    feature_name_mapping: dict,
    columns_of_interest: list = None,
    figsize=(5, 4),
    cmap="BuPu",
    save_dir: str = None,
    filename: str = "pearson_corr",
    save_formats: list = ["svg", "pdf", "png"],
    dpi: int = 300,
    title: str = "Pearson Correlation Heatmap"
):
    """
    Plot and optionally save a Pearson correlation heatmap for specified features.

    Parameters:
    - df: pd.DataFrame
        Input DataFrame containing the data.
    - feature_name_mapping: dict
        Dictionary mapping feature column names to LaTeX-style labels.
    - columns_of_interest: list, optional
        List of columns to include in the correlation matrix. Defaults to all keys of feature_name_mapping.
    - figsize: tuple, optional
        Size of the figure.
    - cmap: str, optional
        Colormap for the heatmap.
    - save_dir: str, optional
        Directory path to save the figure files. If None, figure won't be saved.
    - filename: str, optional
        Base filename for saving the plot.
    - save_formats: list, optional
        List of formats to save the plot in, e.g., ['png', 'svg', 'pdf'].
    - dpi: int, optional
        Resolution for saved PNG files.
    - title: str, optional
        Title of the plot.
    """

    if columns_of_interest is None:
        columns_of_interest = list(feature_name_mapping.keys())

    # Filter dataframe to columns of interest
    filtered_df = df[columns_of_interest]

    # Compute correlation matrix
    correlation_matrix = filtered_df.corr()

    # Rename rows and columns with LaTeX labels
    correlation_matrix.rename(index=feature_name_mapping, columns=feature_name_mapping, inplace=True)

    # Create mask for upper triangle excluding diagonal
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        cmap=cmap,
        fmt=".2f",
        cbar=True,
        linewidths=0.5,
    )
    plt.title(title)
    plt.tight_layout()

    # Save figure if save_dir is specified
    if save_dir:
        for ext in save_formats:
            save_path = f"{save_dir}/{filename}.{ext}"
            if ext == "png":
                plt.savefig(save_path, format=ext, dpi=dpi)
            else:
                plt.savefig(save_path, format=ext)

    plt.show()




# ─────────────────────────────
# MAE by Metal Plot
# ─────────────────────────────



def plot_mae_by_metal(
    df: pd.DataFrame,
    true_value_col: str,
    pred_prefix: str = "pred_",
    metal_col: str = "M1",
    metal_colors: dict = None,
    n_folds: int = 10,
    figsize: tuple = (6, 3),
    title: str = "Mean Absolute Error by Metal Category",
    save_dir: str = None,
    filename: str = None,
    save_formats: list = None,
    dpi: int = 300
):
    """
    Calculates and plots mean absolute error (MAE) by metal category.

    Parameters:
    - df: pd.DataFrame
        DataFrame containing true values, predictions, and metal categories.
    - true_value_col: str
        Column name of the true target values.
    - pred_prefix: str
        Prefix for prediction columns.
    - metal_col: str
        Column name for the metal category.
    - metal_colors: dict
        Mapping of metal categories to 'rgb(r,g,b)' strings.
    - n_folds: int
        Number of cross-validation folds used to normalize MAE.
    - figsize: tuple
        Figure size for plotting.
    - title: str
        Title of the plot.
    - save_dir: str or None
        Directory path to save the plot. If None, plot is not saved.
    - filename: str or None
        Base filename for saving plots (without extension).
    - save_formats: list or None
        List of formats to save the plot in, e.g. ['png', 'svg', 'pdf'].
    - dpi: int
        Resolution for saving plots in dpi.
    """

    # Helper function to convert 'rgb(r, g, b)' string to tuple
    def rgb_to_tuple(rgb_str):
        rgb_values = rgb_str.strip('rgb()').split(',')
        return tuple(int(value) / 255 for value in rgb_values)

    # Filter prediction columns
    prediction_columns = [col for col in df.columns if col.startswith(pred_prefix)]
    if not prediction_columns:
        raise ValueError(f"No columns found starting with '{pred_prefix}'")

    mae_results = []

    # Calculate MAE per metal category for each prediction column
    for pred_col in prediction_columns:
        df['absolute_error'] = abs(df[true_value_col] - df[pred_col])
        mae_by_category = df.groupby(metal_col)['absolute_error'].mean() / n_folds
        mae_results.append(mae_by_category)

    # Combine MAEs into a DataFrame
    mae_df = pd.concat(mae_results, axis=1)
    mae_df.columns = prediction_columns

    # Calculate mean and std of MAE for each metal
    mae_mean = mae_df.mean(axis=1)
    mae_std = mae_df.std(axis=1)

    # Sort by mean MAE descending
    mae_mean_sorted = mae_mean.sort_values(ascending=False)
    mae_std_sorted = mae_std[mae_mean_sorted.index]

    # Set colors for metals, default gray if not found
    if metal_colors is None:
        metal_colors = {}

    colors = [rgb_to_tuple(metal_colors.get(metal, 'rgb(128, 128, 128)')) for metal in mae_mean_sorted.index]

    # Plot
    plt.figure(figsize=figsize)
    plt.bar(mae_mean_sorted.index, mae_mean_sorted, yerr=mae_std_sorted, capsize=5, color=colors, edgecolor=None)
    plt.title(title, fontsize=14)
    plt.xlabel('Metal Category', fontsize=12)
    plt.ylabel('Mean Absolute Error (MAE)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save plots if requested
    if save_dir and filename and save_formats:
        for fmt in save_formats:
            plt.savefig(f"{save_dir}/{filename}.{fmt}", format=fmt, dpi=dpi)

    plt.show()

# ─────────────────────────────
# MAE by Carbon number Plot
# ─────────────────────────────

def plot_mae_by_category(
    df: pd.DataFrame,
    true_col: str = "Eads",
    group_col: str = "cavity_3",
    n_folds: int = 10,
    save_path: str = None,
    save_formats: list = ["png", "svg", "pdf"],
    dpi: int = 300,
    figsize=(6, 3),
    title: str = "Mean Absolute Error by Category"
):
    """
    Plots the mean absolute error (MAE) with error bars for each group (e.g., cavity or metal type),
    based on multiple prediction columns starting with 'pred_'.

    Parameters:
    - df (pd.DataFrame): DataFrame containing true values, predictions, and grouping column.
    - true_col (str): Column name of the ground truth values.
    - group_col (str): Column to group by (e.g., 'M1' for metal or 'cavity_3').
    - n_folds (int): Number of cross-validation folds (for averaging MAE).
    - save_path (str): Directory path to save the figures (optional).
    - save_formats (list): File formats to save the plot in (optional).
    - dpi (int): Resolution of the saved image files.
    - figsize (tuple): Size of the figure.
    - title (str): Plot title.
    """

    # Get prediction columns
    prediction_columns = [col for col in df.columns if col.startswith('pred_')]
    mae_results = []

    # Calculate MAE per fold and category
    for pred_col in prediction_columns:
        df['absolute_error'] = abs(df[true_col] - df[pred_col])
        mae_by_category = df.groupby(group_col)['absolute_error'].mean() / n_folds
        mae_results.append(mae_by_category)

    # Combine into DataFrame
    mae_df = pd.concat(mae_results, axis=1)
    mae_df.columns = prediction_columns
    mae_mean = mae_df.mean(axis=1)
    mae_std = mae_df.std(axis=1)

    # Sort by mean MAE
    mae_mean_sorted = mae_mean.sort_values(ascending=False)
    mae_std_sorted = mae_std[mae_mean_sorted.index]

    # Color map
    norm = plt.Normalize(vmin=mae_mean_sorted.min(), vmax=mae_mean_sorted.max())
    cmap = plt.get_cmap('Purples')
    colors = [cmap(norm(mae)) for mae in mae_mean_sorted]

    # Plot
    plt.figure(figsize=figsize)
    plt.bar(
        mae_mean_sorted.index, mae_mean_sorted,
        yerr=mae_std_sorted, capsize=5,
        color=colors, edgecolor=None
    )
    plt.title(title, fontsize=14)
    plt.xlabel(group_col.replace("_", " ").title(), fontsize=12)
    plt.ylabel("Mean Absolute Error (MAE)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plots if path is provided
    if save_path:
        for ext in save_formats:
            plt.savefig(f"{save_path}/mae_by_{group_col}.{ext}", format=ext, dpi=dpi)

    plt.show()

# ──────────────────────────────────────────────────────────
# Stability-Synergistic Effect Trade-offs Plot
# ──────────────────────────────────────────────────────────

def plot_stability_adsorption(df, save_path=None, show_plot=True):

    """
    Plots a stability vs. adsorption energy trade-off graph for catalyst systems.

    This function visualizes the trade-off between predicted adsorption energies 
    (`y_pred`) and interaction energies (`Eint`) from a dataframe. It highlights 
    specific catalyst systems with colored edge outlines and marks the most stable 
    systems (with minimum `y_pred` and `Eint`) within the stable region 
    (where both values are negative).

    Parameters:
    ----------
    df : pandas.DataFrame
        A dataframe containing at least the following columns:
        - 'y_pred': predicted adsorption energy
        - 'Eint': interaction energy
        - 'system_dacs': identifier string for each catalyst system

    save_path : str, optional
        If provided, the plot will be saved to the specified path (without extension).
        The plot will be saved in SVG, PNG, and PDF formats.

    show_plot : bool, default=True
        If True, the plot will be displayed using matplotlib.
        If False, the plot will be closed after saving (useful for batch runs).

    Returns:
    -------
    None
    """
    # Define stable region
    stable_df = df[(df['y_pred'] < 0) & (df['Eint'] < 0)]

    # Identify minimum points
    min_y_pred_row = stable_df.loc[stable_df['y_pred'].idxmin()]
    min_Eint_row = stable_df.loc[stable_df['Eint'].idxmin()]
    min_y_pred_x, min_y_pred_y = min_y_pred_row['y_pred'], min_y_pred_row['Eint']
    min_Eint_x, min_Eint_y = min_Eint_row['y_pred'], min_Eint_row['Eint']

    # Color assignment logic
    def assign_color(row):
        if row['y_pred'] < 0 and row['Eint'] < 0:
            return (0.5, 0.0, 0.5)
        elif row['y_pred'] > 0 and row['Eint'] > 0:
            return (0.5, 0.5, 0.5)
        elif row['y_pred'] < 0 and row['Eint'] > 0:
            return (0.0, 0.5, 0.5)
        else:
            return (0.0, 0.0, 0.5)

    df['color'] = df.apply(assign_color, axis=1)

    # Create plot
    plt.figure(figsize=(8, 4))
    plt.scatter(df['y_pred'], df['Eint'], c=df['color'], s=50, alpha=0.7)

    # Highlight specific points
    highlight_configs = {
        'din4_x2_black': [
            'Co_Fe_N_din4_x2_c1_b', 'Fe_Fe_N_din4_x2_c2_f', 'Ni_Fe_N_din4_x2_c2_f',
            'Fe_Fe_N_din4_x2_c0', 'Fe_Fe_N_din4_x2_c1_b', 'Co_Fe_N_din4_x2_c0',
            'Ni_Fe_N_din4_x2_c1_b', 'Ni_Fe_N_din4_x2_c0'
        ],
        'din6_as_red': [
            'Ni_Fe_N_din6_as_c3_253', 'Ni_Fe_N_din6_as_c3_235', 'Ni_Fe_N_din6_as_c3_254',
            'Ni_Fe_N_din6_as_c3_125', 'Zn_Fe_N_din6_as_c3_254', 'Zn_Fe_N_din6_as_c3_125',
            'Pt_Fe_N_din6_as_c3_235', 'Pt_Fe_N_din6_as_c3_253'
        ],
        'din6_as_magenta': [
            'Ni_Fe_N_din6_as_c3_236', 'Ni_Fe_N_din6_as_c5_12345'
        ],
        'din6_s_black': [
            'Pt_Fe_C_din6_s_c6', 'Pd_Fe_C_din6_s_c6', 'Ni_Fe_C_din6_s_c6',
            'Zn_Fe_C_din6_s_c6', 'Fe_Fe_C_din6_s_c6', 'Rh_Fe_C_din6_s_c6',
            'Co_Fe_C_din6_s_c6', 'Ir_Fe_C_din6_s_c6'
        ]
    }

    edgecolor_map = {
        'din4_x2_black': 'black',
        'din6_as_red': 'red',
        'din6_as_magenta': 'magenta',
        'din6_s_black': 'black'
    }

    for group, systems in highlight_configs.items():
        for system in systems:
            specific_point = df[df['system_dacs'] == system]
            if not specific_point.empty:
                plt.scatter(
                    specific_point['y_pred'], specific_point['Eint'],
                    facecolor=specific_point['color'].values[0],
                    edgecolor=edgecolor_map[group], s=50, linewidth=1.5, label=system
                )

    # Highlight minimum points
    plt.scatter(min_y_pred_x, min_y_pred_y, color='blue', edgecolor='blue', s=50, linewidth=1.5)
    plt.scatter(min_Eint_x, min_Eint_y, color='blue', edgecolor='blue', s=50, linewidth=1.5)

    # Axes lines
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

    # Labels and limits
    plt.xlabel("$E^{RFR}_{\mathrm{ads}}$ / eV", fontsize=14)
    plt.ylabel("$E_{\mathrm{int}}$ / eV", fontsize=14)
    plt.xlim(-7.5, 2.5)
    plt.ylim(-3, 7)

    # Save or show plot
    plt.tight_layout()
    if save_path:
        for ext in ['svg', 'png', 'pdf']:
            plt.savefig(f"{save_path}.{ext}", format=ext, dpi=300 if ext == 'png' else None)
    if show_plot:
        plt.show()
    else:
        plt.close()


# ─────────────────────────────────────────────────────────────────────
# Correlation plot of the a selected feature and the adsorption energy
# ─────────────────────────────────────────────────────────────────────


def plot_feature_vs_prediction(din4_x2, din6_as, din6_s, 
                               regression_feature='fermi_energy_cavity',
                               hue_feature='cavity_v2',
                               save_path=None, show_plot=True):
    """
    Plots the relationship between a selected regression feature and predicted adsorption energies
    across three datasets (din4_x2, din6_as, din6_s), with annotation and color-coding by another feature.

    Parameters:
        din4_x2 (pd.DataFrame): DataFrame for the din4_x2 dataset.
        din6_as (pd.DataFrame): DataFrame for the din6_as dataset.
        din6_s (pd.DataFrame): DataFrame for the din6_s dataset.
        regression_feature (str): Column name to plot on the x-axis.
        hue_feature (str): Column name used for coloring points.
        save_path (str): Path to save the figure (without extension). Saves as .svg, .png, and .pdf.
        show_plot (bool): Whether to display the plot interactively.
    """

    # Validate feature existence
    required_cols = [regression_feature, 'y_pred', 'M1', hue_feature]
    for df, name in zip([din4_x2, din6_as, din6_s], ['din4_x2', 'din6_as', 'din6_s']):
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' is missing in dataset '{name}'")

    # Add dataset identifiers
    din4_x2 = din4_x2.copy()
    din6_as = din6_as.copy()
    din6_s = din6_s.copy()
    din4_x2['Dataset'] = 'din4_x2'
    din6_as['Dataset'] = 'din6_as'
    din6_s['Dataset'] = 'din6_s'

    # Combine datasets
    combined_data = pd.concat([din4_x2, din6_as, din6_s])

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.5))

    # Default axis limits (can be extended or parameterized)
    x_limits = {'din4_x2': None, 'din6_as': None, 'din6_s': None}
    y_limits = {'din4_x2': None, 'din6_as': None, 'din6_s': None}

    for ax, dataset_name in zip(axes, ['din4_x2', 'din6_as', 'din6_s']):
        dataset_data = combined_data[combined_data['Dataset'] == dataset_name]

        sns.scatterplot(
            data=dataset_data,
            x=regression_feature,
            y='y_pred',
            ax=ax,
            hue=hue_feature,
            palette='viridis',
            s=100,
            alpha=1,
            legend=None
        )

        # Set axis labels and limits
        ax.set_xlabel(regression_feature.replace('_', ' ').capitalize() + ' / eV', fontsize=12)
        ax.set_ylabel("$E^{RFR}_{\mathrm{ads}}$ / eV", fontsize=12)
        ax.grid(False)
        if x_limits[dataset_name]: ax.set_xlim(x_limits[dataset_name])
        if y_limits[dataset_name]: ax.set_ylim(y_limits[dataset_name])

        # Annotate with M1 values
        for _, row in dataset_data.iterrows():
            ax.text(
                row[regression_feature],
                row['y_pred'],
                str(row['M1']),
                fontsize=9,
                ha='left',
                va='top',
                color='black'
            )

    plt.tight_layout()

    # Save plot
    if save_path:
        for ext in ['svg', 'png', 'pdf']:
            plt.savefig(f"{save_path}.{ext}", format=ext, dpi=300 if ext == 'png' else None)

    if show_plot:
        plt.show()
    else:
        plt.close()


# ─────────────────────────────────────────────────────────────────────
# Parity plots for Dual-Metal LOGOCV
# ─────────────────────────────────────────────────────────────────────



def plot_rfr_mlogocv_results(datasets, data_error, metal_colors, output_path=None):
    """
    Plots actual vs predicted adsorption energies for multiple metals using RFR model results.
    
    Parameters:
        datasets (dict): Dictionary of DataFrames with metal names as keys. Each DataFrame must have
                         columns 'y' (true values) and 'predicted_average' (model predictions).
        data_error (DataFrame): DataFrame with error metrics (mae_tests, rmse_tests, rsquared_tests)
                                for each metal.
        metal_colors (dict): Dictionary mapping each metal to an RGB tuple for plotting.
        output_path (str or None): Directory to save plots in .png, .pdf, and .svg. If None, no files are saved.
    """
    error_values = data_error[['Metal', 'mae_tests', 'rmse_tests', 'rsquared_tests']]
    
    # Calculate global min and max for axes
    global_min = min(min(data['y'].min(), data['pred_00000'].min()) for data in datasets.values())
    global_max = max(max(data['y'].max(), data['pred_00000'].max()) for data in datasets.values())
    
    # Set grid dimensions based on number of metals
    num_metals = len(datasets)
    num_rows = 3
    num_cols = 5
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, (metal, data) in enumerate(datasets.items()):
        ax = axes[idx]

        # Compute prediction error
        data['error'] = data['y'] - data['predicted_average']

        # Scatter plot
        sns.scatterplot(
            data=data, x='y', y='predicted_average', ax=ax,
            s=120, alpha=1.0, edgecolor='black', color=metal_colors[metal]
        )

        # Diagonal line
        ax.plot([global_min, global_max], [global_min, global_max], color='black', linestyle='--')

        # Error values for legend
        err = error_values[error_values['Metal'] == metal].iloc[0]
        legend_text = f"$R^2$: {err['rsquared_tests']:.2f} \nRMSE: {err['rmse_tests']:.2f}\nMAE: {err['mae_tests']:.2f}"

        ax.legend(
            labels=[legend_text], loc='upper left', fontsize=10,
            title_fontsize=12, borderpad=1.0, handlelength=1.5,
            labelspacing=1.2, frameon=True, framealpha=0.8
        )

        ax.set_title(f"{metal} - Fe", fontsize=14)
        ax.set_xlabel("E$_{ads}^{DFT}$", fontsize=12)
        ax.set_ylabel("E$_{ads}^{RFR}$", fontsize=12)
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2f}'.format(x)))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2f}'.format(x)))

        # Inset plot
        inset_ax = ax.inset_axes([0.60, 0.12, 0.35, 0.35])
        sns.histplot(data['error'], bins=20, kde=True, color=metal_colors[metal], ax=inset_ax)
        inset_ax.set_xlabel("Error distribution", fontsize=8)
        inset_ax.set_ylabel("Density", fontsize=8)
        inset_ax.tick_params(axis='both', labelsize=8)
        inset_ax.set_xlim(-0.6, 0.6)

    # Hide unused axes
    for i in range(num_metals, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    # Save if output path is provided
    if output_path:
        plt.savefig(f'{output_path}/rfr_mlogocv_dacs.svg', format='svg')
        plt.savefig(f'{output_path}/rfr_mlogocv_dacs.png', format='png', dpi=300)
        plt.savefig(f'{output_path}/rfr_mlogocv_dacs.pdf', format='pdf')

    plt.show()


# ─────────────────────────────────────────────────────────────────────
# Boxplot Adsorption energy 
# ─────────────────────────────────────────────────────────────────────

def plot_categorical_energy_boxplot(
    df, 
    x_col, 
    color_map, 
    y_col='y', 
    hover_col='System', 
    save_dir=None,
    x_title=None,
    y_title=None,
    file_name=None
):
    """
    Create a boxplot of energy values grouped by a categorical column, colored by color_map.
    Adds scatter points with hover information.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        x_col (str): Column name for x-axis categories.
        color_map (dict): Dictionary mapping category names to colors.
        y_col (str): Name of the column with energy values to plot (default 'y').
        hover_col (str): Column to use for hover text on scatter points (default 'System').
        save_dir (str or None): Directory to save plots. If None, plots are not saved.
        x_title (str or None): X-axis title. If None, uses x_col.
        y_title (str or None): Y-axis title. If None, uses y_col.
        file_name (str or None): Base filename to save plots. If None, uses x_col + '_energy_boxplot'.
        
    Returns:
        fig (plotly.graph_objs._figure.Figure): The generated Plotly figure.
    """
    # Map colors
    df['color'] = df[x_col].map(color_map)

    # Create boxplot
    fig = px.box(df, x=x_col, y=y_col, color=x_col, color_discrete_map=color_map)

    # Add scatter points
    fig.add_scatter(
        x=df[x_col],
        y=df[y_col],
        mode='markers',
        marker=dict(color=df['color']),
        hovertext=df[hover_col],
        showlegend=False
    )

    # Update layout
    fig.update_layout(
        xaxis_title=x_title or x_col,
        yaxis_title=y_title or y_col
    )

    # Save if directory given
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        base_name = file_name or f"{x_col}_energy_boxplot"
        base_path = os.path.join(save_dir, base_name)
        fig.write_html(f"{base_path}.html")
        pio.write_image(fig, f"{base_path}.png")
        pio.write_image(fig, f"{base_path}.pdf")

    return fig


# ─────────────────────────────────────────────────────────────────────
# Histogram of the adsorption energies by metal type
# ─────────────────────────────────────────────────────────────────────

def plot_energy_histograms_by_metal(
    df, 
    metal_col, 
    energy_col, 
    color_map, 
    ncols=3, 
    bins=50, 
    figsize=(12, 9), 
    save_dir=None,
    file_name=None
):
    """
    Plot histograms of energy values grouped by metal types with mean value lines.
    Optionally save the figure as PNG and PDF with 300 dpi.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        metal_col (str): Column name for metal types (e.g. 'M1').
        energy_col (str): Column name for energy values to plot (e.g. 'E_dft_M1M2').
        color_map (dict): Dictionary mapping metal types to colors.
        ncols (int): Number of columns in subplot grid.
        bins (int): Number of bins for histograms.
        figsize (tuple): Figure size for the entire plot.
        save_dir (str or None): Directory path to save the figure. If None, figure is not saved.
        file_name (str or None): Filename (without extension) for saving the figure.
                                 Must be provided if save_dir is provided.
    """
    

    unique_metal_values = df[metal_col].unique()
    nrows = int(np.ceil(len(unique_metal_values) / ncols))
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten() if len(unique_metal_values) > 1 else [axes]

    for i, metal_type in enumerate(unique_metal_values):
        ax = axes[i]
        subset_df = df[df[metal_col] == metal_type]

        sns.histplot(
            data=subset_df,
            x=energy_col,
            bins=bins,
            color=color_map.get(metal_type, 'gray'),
            ax=ax,
            kde=False
        )
        
        mean_value = subset_df[energy_col].mean()
        ax.axvline(
            x=mean_value,
            color=color_map.get(metal_type, 'gray'),
            linestyle='--',
            linewidth=2
        )
        ax.text(
            mean_value,
            ax.get_ylim()[1] * 0.9,
            f'{metal_type}',
            color=color_map.get(metal_type, 'gray'),
            ha='right' if mean_value > subset_df[energy_col].mean() else 'left',
            fontsize=12
        )
        
        ax.set_title(f'{metal_type}')
        ax.set_xlabel(energy_col)
        ax.set_ylabel('Count')

    # Remove unused axes
    for j in range(len(unique_metal_values), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.suptitle(f'Histogram of {energy_col} by {metal_col}', y=1.02)

    if save_dir is not None and file_name is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{save_path}.pdf", bbox_inches='tight')

    plt.show()



# ─────────────────────────────────────────────────────────────────────
# Donut plot
# ─────────────────────────────────────────────────────────────────────

def plot_donut_chart(
    sizes,
    labels,
    colors=None,
    title=None,
    save_path=None
):
    """
    Plot a donut chart from given sizes and labels, and optionally save it.

    Parameters:
        sizes (list of float): Percentages or values for each category.
        labels (list of str): Corresponding labels for each section.
        colors (list of str, optional): Colors for the sections. Defaults to matplotlib's default.
        title (str, optional): Title for the chart.
        save_path (str, optional): Path (without extension) to save the figure as .png, .pdf, and .svg.
    """
    plt.figure(figsize=(6, 6))
    plt.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct='%1.2f%%',
        wedgeprops={'edgecolor': 'white'}
    )
    
    # Draw center white circle to make it a donut
    center_circle = plt.Circle((0, 0), 0.70, fc='white')
    plt.gca().add_artist(center_circle)

    if title:
        plt.title(title)

    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f'{save_path}.svg', format='svg')
        plt.savefig(f'{save_path}.png', format='png', dpi=300)
        plt.savefig(f'{save_path}.pdf', format='pdf')

    plt.show()
