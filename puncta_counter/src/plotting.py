import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

from bokeh.models.annotations import Title
from bokeh.models import Plot, ColumnDataSource, Ellipse, Grid, LinearAxis, Text
from bokeh.io import export_png


# Functions
# # save_fig_as_png
# # save_plot_as_png
# # plot_single_scatter


def save_fig_as_png(fig, filepath, width=1200, height=800, scale=1, engine="kaleido"):
    """For plotly
    Make sure file extension is ".png"
    """
    if os.path.sep in filepath:
        os.makedirs(os.path.sep.join(str(filepath).split(os.path.sep )[:-1]), exist_ok=True)
    fig.write_image(filepath, width=width, height=height, scale=scale, engine=engine)


def save_plot_as_png(plot, filepath):
    """For Bokeh
    """
    if os.path.sep in filepath:
        os.makedirs(os.path.sep.join(str(filepath).split(os.path.sep )[:-1]), exist_ok=True)
    export_png(plot, filename=filepath)


def plot_single_scatter(
    df, x, y, title=None,
    error_y=None, error_y_minus=None,
    xlabel=None, ylabel=None,
    xmin=None, xmax=None,
    ymin=None, ymax=None,
    log_x=False,
    mode='markers',
    color_discrete_sequence=px.colors.qualitative.G10,
):
    """Plots a single scatter function
    """
    
    # ----------------------------------------------------------------------
    # X Range
    
    if not log_x:
        xaxis_type = 'linear'
        # x_range
        if not xmin:
            xmin = df[x].min()
        if not xmax:
            xmax = df[x].max()
        xrange = [xmin - (xmax-xmin) * 0.05, xmax + (xmax-xmin) * 0.05]
    else:
        xaxis_type = 'log'
        xrange = [None, None]
    
    
    # ----------------------------------------------------------------------
    # Y Range
    
    if error_y:
        max_error_y_plus = df[error_y].max()
    else:
        max_error_y_plus = 0          
    if error_y_minus:
        max_error_y_minus = df[error_y_minus].max()
    else:
        max_error_y_minus = 0      
    if not ymin:
        ymin = df[y].min()-max_error_y_minus
    if not ymax:
        ymax = df[y].max()+max_error_y_plus   
    yrange = [ymin - (ymax-ymin) * 0.1, ymax + (ymax-ymin) * 0.1]
    
    
    # ----------------------------------------------------------------------
    # Figure
    
    fig = go.Figure()
    scatter = px.scatter(
        df, x=x, y=y, error_y=error_y, error_y_minus=error_y_minus,
        color_discrete_sequence=px.colors.qualitative.G10,
    )
    fig.add_trace(scatter.data[0])

    
    # ----------------------------------------------------------------------
    # Layout
    
    fig.layout.update(
        title=go.layout.Title(text=title),
        xaxis={'title_text': xlabel,
               'showgrid': True,
               'gridcolor': '#E4EAF2', 
               'range': xrange,
               'type': xaxis_type
              },
        yaxis={'title_text': ylabel,
               'showgrid': True, 'gridcolor': '#E4EAF2', 'zeroline': False,
               'range': yrange},
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='closest',
        showlegend=False,
    )
    
    fig.data[0].update(mode=mode)
    fig.update_traces(marker=dict(size=8))

    return fig


def plot_circle_puncta_using_plotly(puncta_summary, puncta, image_number, title):

    # get nuclei boundaries
    shapes = list(
        puncta_summary.loc[
            (puncta_summary['image_number']==image_number)
            , ["bounding_box_min_x",
              "bounding_box_max_x",
              "bounding_box_min_y",
              "bounding_box_max_y",]
        ]
        .rename(columns={
                    "bounding_box_min_x": "x0",
                    "bounding_box_max_x": "x1",
                    "bounding_box_min_y": "y0",
                    "bounding_box_max_y": "y1",}
               )
        .apply(lambda x: {**{"type": "circle", 'xref':"x", 'yref':"y", 'line':{'width':1.5}}, **dict(x)}, axis=1)
    )

    # plot puncta
    fig = plot_single_scatter(
        puncta[puncta['image_number']==image_number].copy(),
        x='center_x',
        y='center_y',
        title=title,
        xlabel='x',
        ylabel='y'
    )

    fig.layout.update(
        xaxis = {'range': [-50, 1250], 'constrain': "domain"},
        yaxis = {'range': [1050, -50], 'scaleanchor': 'x', 'scaleratio': 1},
        shapes=shapes,
        height=700,
    )
    fig.update_traces(
        marker=dict(size=3)
    )

    return fig


def plot_ellipse_using_bokeh(puncta_summary, puncta, image_number, title):

    # subset by image
    nuclei_subset = puncta_summary[puncta_summary['image_number']==image_number].copy()
    puncta_subset = pd.merge(
        left=nuclei_subset[["image_number", 'object_number']],
        right=puncta.loc[:, puncta.columns != 'object_number'],
        left_on=["image_number", 'object_number'],
        right_on=['image_number', 'nuclei_object_number'],
        how="left",
    ).dropna(subset=['nuclei_object_number'])


    # add nuclei
    nuclei_subset['angle'] = nuclei_subset['orientation'].apply(lambda x: x/180*np.pi)
    nuclei_source = ColumnDataSource(nuclei_subset.loc[
        # (nuclei_subset['object_number']==2)
        :, ["object_number", "center_x", "center_y", "major_axis_length", "minor_axis_length", "angle"]
    ].rename(
        columns={
            "center_x": "x",
            "center_y": "y",
            "major_axis_length": "h",
            "minor_axis_length": "w"
        }
    ).to_dict("list"))
    nuclei_glyph = Ellipse(x="x", y="y", width="w", height="h", angle='angle', line_color='#FFFFFF', fill_color='#000fff', line_width=1.2)
    text_glyph = Text(x="x", y="y", text="object_number", text_color="white", text_font_size = {'value': '13px'})


    # add puncta
    puncta_subset['angle'] = puncta_subset['orientation'].apply(lambda x: x/180*np.pi)

    puncta_source = ColumnDataSource(puncta_subset[
        ["center_x", "center_y", "major_axis_length", "minor_axis_length", "angle"]
    ].rename(
        columns={
            "center_x": "x",
            "center_y": "y",
            "major_axis_length": "h",
            "minor_axis_length": "w"
        }
    ).to_dict("list"))
    puncta_glyph = Ellipse(x="x", y="y", width="w", height="h", angle='angle', fill_color='#ff2b00', line_alpha=0, )

    puncta_text_source = ColumnDataSource(puncta_summary.loc[
        (puncta_summary['image_number']==image_number)
        & (puncta_summary['object_number'].isin(nuclei_subset['object_number']))
        , ["image_number", "object_number", "center_x", "center_y",]
    ].to_dict("list"))
    puncta_text_glyph = Text(
        x="center_x", y="center_y", text="object_number", text_color="orange", text_font_size = {'value': '13px'}
    )


    # add puncta
    plot = Plot(
        title=Title(text=title),
        width=1000, height=800,
        match_aspect=True,
        toolbar_location=None
    )
    plot.add_glyph(nuclei_source, nuclei_glyph)
    plot.add_glyph(nuclei_source, text_glyph)
    plot.add_glyph(puncta_source, puncta_glyph)
    plot.add_glyph(puncta_text_source, puncta_text_glyph)


    xaxis = LinearAxis()
    plot.add_layout(xaxis, 'above')
    plot.x_range.start = 0
    plot.x_range.end = 1290

    yaxis = LinearAxis()
    plot.add_layout(yaxis, 'left')
    plot.y_range.start = 1000
    plot.y_range.end = 0

    plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
    plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))

    return plot
