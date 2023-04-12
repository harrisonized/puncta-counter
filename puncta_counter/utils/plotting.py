"""General plotting functions in Bokeh
Bokeh was chosen over Plotly because of its ability to plot rotated ellipses
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from bokeh.models.annotations import Title
from bokeh.models import Plot, LinearAxis, Grid, ColumnDataSource, Ellipse, Circle, Scatter, Text
from bokeh.io import export_png


# Functions
# # save_plot_as_png
# # instantiate_plot_bokeh
# # plot_ellipse_using_bokeh
# # plot_circle_using_bokeh
# # plot_scatter_using_bokeh
# # plot_split_violin


def save_plot_as_png(
		plot,
		filepath,
		width=None,  # doesn't work for some reason
		height=None,  # doesn't work for some reason
	):
    """For Bokeh
    """
    if os.path.sep in filepath:
        os.makedirs(os.path.sep.join(str(filepath).split(os.path.sep )[:-1]), exist_ok=True)
    export_png(plot, filename=filepath, width=width, height=height)


def instantiate_plot_bokeh(
    title=None,
    width=1000,
    height=800,
    toolbar_location=None,
    xaxis_position='above',
    yaxis_position='left',
    x_range=[0, 1290],
    y_range=[1000, 0],
):
    plot = Plot(
        title=Title(text=title),
        width=width,
        height=height,
        match_aspect=True,
        toolbar_location=toolbar_location
    )

    # plot.title = Title(text=title)
    xaxis = LinearAxis()
    plot.add_layout(xaxis, xaxis_position)
    plot.x_range.start = x_range[0]
    plot.x_range.end = x_range[1]

    yaxis = LinearAxis()
    plot.add_layout(yaxis, yaxis_position)
    plot.y_range.start = y_range[0]
    plot.y_range.end = y_range[1]

    plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
    plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))
    
    return plot


def plot_ellipse_using_bokeh(
        column_data,
        text_data=None,
        x='x',
        y='y',
        height="height",
        width="width",
        angle='angle',
        angle_units='deg',
        text=None,
        title=None,
        fill_color='#000fff',  # blue
        fill_alpha=1,
        text_color='white',
        line_alpha=1.2,
        line_width=0,
        plot=None,
    ):
    
    """Input a dataframe: df[['x', 'y', 'height', 'width', 'angle', 'text']]
    """
    
    if plot is None:
        plot=instantiate_plot_bokeh()
        
    if title:
        plot.title = Title(text=title)

    column_data_source = ColumnDataSource(column_data)
        
    ellipse_glyph = Ellipse(
        x=x, y=y, width=width,
        height=height,
        angle=angle,
        angle_units=angle_units,
        line_color='#FFFFFF',
        fill_color=fill_color,
        fill_alpha=fill_alpha,
        line_width=line_width,
        line_alpha=line_alpha,
    )
    plot.add_glyph(column_data_source, ellipse_glyph)
    
    if text_data is not None:
        text_data_source = ColumnDataSource(text_data)
        text_glyph = Text(
            x=x, y=y,
            text=text,
            text_color=text_color,
            text_font_size = {'value': '13px'}
        )
        plot.add_glyph(text_data_source, text_glyph)
    
    return plot


def plot_circle_using_bokeh(
        column_data,
        text_data=None,
        x='x',
        y='y',
        size="size",
        text=None,
        title=None,
        fill_color='#000fff',  # blue
        fill_alpha=1,
        text_color='white',
        line_alpha=1.2,
        line_width=0,
        plot=None,
    ):
    
    """Input a dataframe: df[['x', 'y', 'size' 'text']]
    """
    
    if plot is None:
        plot=instantiate_plot_bokeh()
        
    if title:
        plot.title = Title(text=title)

    column_data_source = ColumnDataSource(column_data)
        
    circle_glyph = Circle(
        x=x, y=y,
        size=size,
        line_color='#FFFFFF',
        fill_color=fill_color,
        fill_alpha=fill_alpha,
        line_width=line_width,
        line_alpha=line_alpha,
    )
    plot.add_glyph(column_data_source, circle_glyph)
    
    if text_data is not None:
        text_data_source = ColumnDataSource(text_data)
        text_glyph = Text(
            x=x, y=y,
            text=text,
            text_color=text_color,
            text_font_size = {'value': '13px'}
        )
        plot.add_glyph(text_data_source, text_glyph)
    
    return plot


def plot_scatter_using_bokeh(
        column_data,
        x='x',
        y='y',
        size=4,
        title=None,
        plot=None,
        width=400,
        height=400,
        x_range=[-10, 10],
        y_range=[-10, 10],
    ):
    
    """Input a dataframe: df[['x', 'y', 'size']]
    """
    
    if plot is None:
        plot=instantiate_plot_bokeh(
            width=width,
            height=height,
            x_range=x_range,
            y_range=y_range,
        )
        
    if title:
        plot.title = Title(text=title)

    column_data_source = ColumnDataSource(column_data)
        
    scatter_glyph = Scatter(
        x=x,
        y=y,
        size=size,
    )
    plot.add_glyph(column_data_source, scatter_glyph)
    
    return plot


def plot_split_violin(
        left, right,
        x, y,
        xlabel=None, ylabel=None, title=None,
        left_group_label=None,
        right_group_label=None,
        ymin=None, ymax=None,
        left_color='#2ca02c',  # green
        right_color='#d62728',  # red
        xshowgrid=False, yshowgrid=True,
        show_points=False,
        showlegend=True,
        legend_title=None,
        box_visible=True,
    ):
    """See: https://plotly.com/python/violin/
    """
    
    fig = go.Figure()

    fig.add_trace(go.Violin(x=left[x],
                            y=left[y],
                            legendgroup=left_group_label, scalegroup=left_group_label, name=left_group_label,
                            side='negative', box_visible=box_visible,
                            line_color=left_color)
                 )
    fig.add_trace(go.Violin(x=right[x],
                            y=right[y],
                            legendgroup=right_group_label, scalegroup=right_group_label, name=right_group_label,
                            side='positive', box_visible=box_visible,
                            line_color=right_color)
                 )
    fig.update_traces(meanline_visible=True)
    fig.update_traces(points='all')
    
    fig.layout.update(
        title=go.layout.Title(text=title),
        xaxis={'title_text': xlabel,
               'showgrid': xshowgrid,
               'gridcolor': '#E4EAF2',
               'zeroline': False,
               },
        yaxis={'title_text': ylabel,
               'range': [ymin, ymax],
               'showgrid': yshowgrid,
               'gridcolor': '#E4EAF2',
               'zeroline': False,
               },
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=showlegend,
        legend=dict(title=legend_title),
        hovermode='closest'
    )
    fig.update_layout(violingap=0, violinmode='overlay')
    
    if show_points:
        fig.update_traces(meanline_visible=True)
        fig.update_traces(points='all')
        
    
    return fig
