import base64
import io
import pathlib

import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from PIL import Image
from io import BytesIO
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

import pandas as pd
import plotly.graph_objs as go
import scipy.spatial.distance as spatial_distance

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

data_dict = {
    "mnist_3000": pd.read_csv(DATA_PATH.joinpath("mnist_3000_input.csv")),
    "wikipedia_3000": pd.read_csv(DATA_PATH.joinpath("wikipedia_3000.csv")),
    "Citta_Design": pd.read_csv(
        DATA_PATH.joinpath("Citta_Design.csv"), encoding="ISO-8859-1"
    ),
}

# Import datasets here for running the Local version
IMAGE_DATASETS = "mnist_3000"
WORD_EMBEDDINGS = ("Citta_Design")


with open(PATH.joinpath("demo_intro.md"), "r") as file:
    demo_intro_md = file.read()

with open(PATH.joinpath("demo_description.md"), "r") as file:
    demo_description_md = file.read()


def numpy_to_b64(array, scalar=True):
    # Convert from 0-1 to 0-255
    if scalar:
        array = np.uint8(255 * array)

    im_pil = Image.fromarray(array)
    buff = BytesIO()
    im_pil.save(buff, format="png")
    im_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")

    return im_b64


# Methods for creating components in the layout code
def Card(children, **kwargs):
    return html.Section(children, className="card-style")


def NamedSlider(name, short, min, max, step, val, marks=None):
    if marks:
        step = None
    else:
        marks = {i: i for i in range(min, max + 1, step)}

    return html.Div(
        style={"margin": "25px 5px 30px 0px"},
        children=[
            f"{name}:",
            html.Div(
                style={"margin-left": "5px"},
                children=[
                    dcc.Slider(
                        id=f"slider-{short}",
                        min=min,
                        max=max,
                        marks=marks,
                        step=step,
                        value=val,
                    )
                ],
            ),
        ],
    )


def NamedInlineRadioItems(name, short, options, val, **kwargs):
    return html.Div(
        id=f"div-{short}",
        style={"display": "inline-block"},
        children=[
            f"{name}:",
            dcc.RadioItems(
                id=f"radio-{short}",
                options=options,
                value=val,
                labelStyle={"display": "inline-block", "margin-right": "7px"},
                style={"display": "inline-block", "margin-left": "7px"},
            ),
        ],
    )


def create_layout(app):
    # Actual layout of the app
    return html.Div(
        className="row",
        style={"max-width": "100%", "font-size": "1.5rem", "padding": "0px 0px"},
        children=[
            # Header
            html.Div(
                className="row header",
                id="app-header",
                style={"background-color": "#f9f9f9"},
                children=[
                    html.Div(
                        [
                            html.Img(
                                src=app.get_asset_url("dash-logo.png"),
                                className="logo",
                                id="plotly-image",
                            )
                        ],
                        className="three columns header_img",
                    ),
                    html.Div(
                        [
                            html.H3(
                                "Itai Walk",
                                className="header_title",
                                id="app-title",
                            )
                        ],
                        className="nine columns header_title_container",
                    ),
                ],
            ),
            # Demo Description
            html.Div(
                className="row background",
                id="demo-explanation",
                style={"padding": "50px 45px"},
                children=[
                    html.Div(
                        id="description-text", children=dcc.Markdown(demo_intro_md)
                    ),
                    html.Div(
                        html.Button(id="learn-more-button", children=["Learn More"])
                    ),
                ],
            ),
            # Body
            html.Div(
                className="row background",
                style={"padding": "10px"},
                children=[
                    html.Div(
                        className="three columns",
                        children=[
                            Card(
                                [
                                    dcc.Dropdown(
                                        id="dropdown-dataset",
                                        searchable=False,
                                        clearable=False,
                                        options=[
                                            {
                                                "label": "Citta Design",
                                                "value": "Citta_Design",
                                            },
                                        ],
                                        placeholder="Select a dataset",
                                        value="Citta_Design",
                                    ),
                                    NamedInlineRadioItems(
                                        name="Popularity based size",
                                        short="popsize-display-mode",
                                        options=[
                                            {
                                                "label": " Regular",
                                                "value": "regular",
                                            },
                                            {
                                                "label": " Size by popularity",
                                                "value": "pop",
                                            },
                                        ],
                                        val="regular",
                                    ),
                                    NamedInlineRadioItems(
                                        name="Dimension reduction method",
                                        short="dimredc-method-mode",
                                        options=[
                                            {
                                                "label": " ISO map",
                                                "value": "iso",
                                            },
                                            {
                                                "label": " tSNE",
                                                "value": "tsne",
                                            },
                                        ],
                                        val="iso",
                                    ),
                                    html.Div(
                                        id="div-wordemb-controls",
                                        style={"display": "none"},
                                        children=[
                                            NamedInlineRadioItems(
                                                name="Display Mode",
                                                short="wordemb-display-mode",
                                                options=[
                                                    {
                                                        "label": " Regular",
                                                        "value": "regular",
                                                    },
                                                    {
                                                        "label": " Top Neighbors",
                                                        "value": "neighbors_mode",
                                                    },
                                                ],
                                                val="regular",
                                            ),
                                            dcc.Dropdown(
                                                id="dropdown-word-selected",
                                                placeholder="Select an item to display its neighbors",
                                                style={"background-color": "#f2f3f4"},
                                            ),
                                            NamedSlider(
                                                name="Number of Neighbors",
                                                short="neighbours_num",
                                                min=5,
                                                max=50,
                                                step=None,
                                                val=5,
                                                marks={
                                                    i: str(i) for i in [i for i in (range(5,51,5))]
                                                },
                                            ),
                                        ],

                                    ),
                                ]
                            )
                        ],
                    ),
                    html.Div(
                        className="six columns",
                        children=[
                            dcc.Graph(id="graph-3d-plot-tsne", style={"height": "98vh"})
                        ],
                    ),
                    html.Div(
                        className="three columns",
                        id="euclidean-distance",
                        children=[
                            Card(
                                style={"padding": "5px"},
                                children=[
                                    html.Div(
                                        id="div-plot-click-message",
                                        style={
                                            "text-align": "center",
                                            "margin-bottom": "7px",
                                            "font-weight": "bold",
                                        },
                                    ),
                                    html.Div(
                                        [
                                        html.Div(id="div-plot-click-title"),
                                            html.Img(
                                                src=app.get_asset_url("dash-logo.png"),
                                                className="logo",
                                                id="xxx",
                                            )

                                        ]
                                    ),
                                    html.Div(id="div-plot-click-wordemb"),
                                ],
                            )
                        ],
                    ),
                ],
            ),
        ],
    )

def get_image_url(self, dataset, path):
    data_url = [
        "images",
        str(dataset),
        path,
    ]
    full_path = PATH.joinpath(*data_url)
    return full_path


def demo_callbacks(app):
    def generate_figure_image(groups, layout):
        data = []

        for idx, val in groups:
            scatter = go.Scatter3d(
                name=idx,
                x=val["x"],
                y=val["y"],
                z=val["z"],
                text=[idx for _ in range(val["x"].shape[0])],
                textposition="top center",
                mode="markers",
                marker=dict(size=3, symbol="circle"),
            )
            data.append(scatter)

        figure = go.Figure(data=data, layout=layout)

        return figure

    # Scatter Plot of the t-SNE datasets
    def generate_figure_word_vec(
        embedding_df, layout, wordemb_display_mode, popsize_display_mode, selected_word, neighbours_num, dataset
    ):

        try:
            point_size= 3
            if not popsize_display_mode == "regular":
                point_size = embedding_df["popularity"].apply(np.sqrt)
            # Regular displays the full scatter plot with only circles
            if wordemb_display_mode == "regular":
                plot_mode = "markers"


            # Nearest Neighbors displays only the K nearest neighbors of the selected_word, in text rather than circles
            elif wordemb_display_mode == "neighbors_mode":
                if not selected_word:
                    return go.Figure()
                plot_mode = "text+markers"

                # Get the nearest neighbors indices using Euclidean distance
                vector = data_dict[dataset].set_index("0")
                selected_vec = vector.loc[selected_word]
                if isinstance(selected_vec, pd.core.frame.DataFrame):
                    selected_vec= selected_vec.iloc[0,:]

                def compare_pd(vector):
                    return spatial_distance.euclidean(vector, selected_vec)

                # vector.apply takes compare_pd function as the first argument
                distance_map = vector.apply(compare_pd, axis=1)
                neighbors_idx = distance_map.sort_values()[:neighbours_num].index

                # Select those neighbors from the embedding_df
                embedding_df = embedding_df.loc[neighbors_idx]


            scatter = [go.Scatter3d(
                name=i,
                x=embedding_df[embedding_df["category"] == i]["x"],
                y=embedding_df[embedding_df["category"] == i]["y"],
                z=embedding_df[embedding_df["category"] == i]["z"],
                text=[str[0:min(20, len(str))] for str in embedding_df[embedding_df["category"] == i].index], #embedding_df.index,#
                hovertext= embedding_df[embedding_df["category"] == i].index,
                textposition="middle center",
                showlegend=True,
                mode=plot_mode,
                marker=dict(size=point_size, symbol="circle"),
            ) for i in embedding_df["category"].unique()]

            figure = go.Figure(data= scatter, layout=layout)

            return figure
        except KeyError as error:
            print(selected_word)
            print(error)
            raise PreventUpdate

    # Callback function for the learn-more button
    @app.callback(
        [
            Output("description-text", "children"),
            Output("learn-more-button", "children"),
        ],
        [Input("learn-more-button", "n_clicks")],
    )
    def learn_more(n_clicks):
        # If clicked odd times, the instructions will show; else (even times), only the header will show
        if n_clicks == None:
            n_clicks = 0
        if (n_clicks % 2) == 1:
            n_clicks += 1
            return (
                html.Div(
                    style={"padding-right": "15%"},
                    children=[dcc.Markdown(demo_description_md)],
                ),
                "Close",
            )
        else:
            n_clicks += 1
            return (
                html.Div(
                    style={"padding-right": "15%"},
                    children=[dcc.Markdown(demo_intro_md)],
                ),
                "Learn More",
            )

    @app.callback(
        Output("div-wordemb-controls", "style"), [Input("dropdown-dataset", "value")]
    )
    def show_wordemb_controls(dataset):
        if dataset in WORD_EMBEDDINGS:
            return None
        else:
            return {"display": "none"}

    @app.callback(
        Output("dropdown-word-selected", "disabled"),
        [Input("radio-wordemb-display-mode", "value")],
    )
    def disable_word_selection(mode):
        return not mode == "neighbors_mode"

    @app.callback(
        Output("dropdown-word-selected", "options"),
        [Input("dropdown-dataset", "value")],
    )
    def fill_dropdown_word_selection_options(dataset):
        if dataset in WORD_EMBEDDINGS:
            return [
                {"label": i, "value": i} for i in data_dict[dataset].iloc[:, 0].tolist()
            ]
        else:
            return []

    @app.callback(
        Output("graph-3d-plot-tsne", "figure"),
        [
            Input("dropdown-dataset", "value"),
            Input("radio-popsize-display-mode", "value"),
            Input("radio-dimredc-method-mode", "value"),
            Input("dropdown-word-selected", "value"),
            Input("radio-wordemb-display-mode", "value"),
            Input("slider-neighbours_num", "value"),
        ],
    )
    def display_3d_scatter_plot(
        dataset,
        popsize_display_mode,
        method,
        selected_word,
        wordemb_display_mode,
        neighbours_num
    ):
        if dataset:
            path = f"demo_embeddings/{dataset}/method_{method}/"

            try:

                data_url = [
                    "demo_embeddings",
                    str(dataset),
                    "method_" + method,
                    "data.csv",
                ]
                full_path = PATH.joinpath(*data_url)
                embedding_df = pd.read_csv(
                    full_path, index_col=0, encoding="ISO-8859-1"
                )

            except FileNotFoundError as error:
                print(
                    error,
                    "\nThe dataset was not found. Please generate it using generate_demo_embeddings.py",
                )
                return go.Figure()

            # Plot layout
            axes = dict(title="", showgrid=True, zeroline=False, showticklabels=False)

            layout = go.Layout(
                margin=dict(l=0, r=0, b=0, t=0),
                scene=dict(xaxis=axes, yaxis=axes, zaxis=axes),
            )

            # For Image datasets
            if dataset in IMAGE_DATASETS:
                embedding_df["label"] = embedding_df.index

                groups = embedding_df.groupby("label")
                figure = generate_figure_image(groups, layout)

            # Everything else is word embeddings
            elif dataset in WORD_EMBEDDINGS:
                figure = generate_figure_word_vec(
                    embedding_df=embedding_df,
                    layout=layout,
                    wordemb_display_mode=wordemb_display_mode,
                    popsize_display_mode= popsize_display_mode,
                    selected_word=selected_word,
                    neighbours_num= neighbours_num,
                    dataset=dataset,
                )

            else:
                figure = go.Figure()

            return figure

    @app.callback(
            Output("div-plot-click-title", "children"),
        [
            Input("graph-3d-plot-tsne", "clickData"),
            Input("dropdown-dataset", "value"),
        ]
    )
    def display_click_title(
        clickData, dataset
    ):
        if dataset in WORD_EMBEDDINGS and clickData:
            selected_word = clickData["points"][0]["hovertext"]

            try:
                # Get the nearest neighbors indices using Euclidean distance
                vector = data_dict[dataset].set_index("0")
                selected_vec = vector.loc[selected_word]
                # very ugly workaround - just until we will have unique product Id
                if isinstance(selected_vec, pd.core.frame.DataFrame):
                    selected_vec= selected_vec.iloc[0,:]
                ###
                path = f"images/{dataset}/"

                data_url = [
                    "images",
                    str(dataset),
                    "citta-logo.png",
                ]
                full_path = PATH.joinpath(*data_url)
                return html.H4(
                                selected_word,
                                className="header_title",
                                id="app-title",
                        )

            except KeyError as error:
                raise PreventUpdate
        return None

#redundant - should be mergred into one function with display_click_title, couldn;t figure out how to output two outputs
    # @app.callback(
    #         Output("div-plot-click-image", "children"),
    #     [
    #         Input("graph-3d-plot-tsne", "clickData"),
    #         Input("dropdown-dataset", "value"),
    #     ]
    # )
    # def display_click_image(
    #     clickData, dataset
    # ):
    #     if dataset in WORD_EMBEDDINGS and clickData:
    #         selected_word = clickData["points"][0]["hovertext"]
    #
    #         try:
    #             # Get the nearest neighbors indices using Euclidean distance
    #             vector = data_dict[dataset].set_index("0")
    #             selected_vec = vector.loc[selected_word]
    #             # very ugly workaround - just until we will have unique product Id
    #             if isinstance(selected_vec, pd.core.frame.DataFrame):
    #                 selected_vec= selected_vec.iloc[0,:]
    #             ###
    #             path = f"images/{dataset}/"
    #
    #             data_url = [
    #                 "images",
    #                 str(dataset),
    #                 "citta-logo.png",
    #             ]
    #             full_path = PATH.joinpath(*data_url)
    #             return html.Img(
    #                         src=full_path ,
    #                         style={"height": "25vh", "display": "block", "margin": "auto"},
    #             )
    #
    #         except KeyError as error:
    #             raise PreventUpdate
    #     return None


    @app.callback(
        Output("div-plot-click-wordemb", "children"),
        [Input("graph-3d-plot-tsne", "clickData"), Input("dropdown-dataset", "value")],
    )
    def display_click_word_neighbors(clickData, dataset):
        if dataset in WORD_EMBEDDINGS and clickData:
            selected_word = clickData["points"][0]["hovertext"]

            try:
                return display_click_word_neighbours_with_word(dataset, selected_word)
            except KeyError as error:
                raise PreventUpdate
        return None

    def display_click_word_neighbours_with_word(dataset, selected_word):
        # Get the nearest neighbors indices using Euclidean distance
        vector = data_dict[dataset].set_index("0")
        selected_vec = vector.loc[selected_word]
        # very ugly workaround - just until we will have unique product Id
        if isinstance(selected_vec, pd.core.frame.DataFrame):
            selected_vec = selected_vec.iloc[0, :]

        def compare_pd(vector):
            return spatial_distance.euclidean(vector, selected_vec)

        # vector.apply takes compare_pd function as the first argument
        distance_map = vector.apply(compare_pd, axis=1)
        num_of_neighbors = 10
        nearest_neighbors = distance_map.sort_values()[1:(num_of_neighbors + 1)].iloc[::-1]
        max_len_of_yaxis_str = 20
        products_titles = [str[0:min(max_len_of_yaxis_str, len(str))] for str in nearest_neighbors.index]
        trace = go.Bar(
            x=nearest_neighbors.values,
            y=products_titles,
            text=products_titles,
            hovertext=nearest_neighbors.index,
            width=0.5,
            orientation="h",
            marker=dict(color="rgb(50, 102, 193)"),
        )
        layout = go.Layout(
            width=400,
            title=f'{num_of_neighbors} nearest neighbors {selected_word}',
            xaxis=dict(title="Euclidean Distance"),
            barmode="overlay",
            # yaxis=dict(title="Product"),
            margin=go.layout.Margin(l=250, r=50, t=35, b=35),
        )
        fig = go.Figure(data=[trace], layout=layout)
        return dcc.Graph(
            id="graph-bar-nearest-neighbors-word",
            figure=fig,
            style={"height": "25vh"},
            config={"displayModeBar": False},
        )

    @app.callback(
        Output("div-plot-click-message", "children"),
        [Input("graph-3d-plot-tsne", "clickData"), Input("dropdown-dataset", "value")],
    )
    def display_click_message(clickData, dataset):
        # Displays message shown when a point in the graph is clicked, depending whether it's an image or word
        if dataset in IMAGE_DATASETS:
            if clickData:
                return "Image Selected"
            else:
                return "Click a data point on the scatter plot to display its corresponding image."

        elif dataset in WORD_EMBEDDINGS:
            if clickData:
                return None
            else:
                return "Click a word on the plot to see its top 10 neighbors."
