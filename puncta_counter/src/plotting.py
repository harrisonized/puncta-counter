import os
import plotly.graph_objs as go
import plotly.express as px


# Functions
# # save_fig_as_png
# # plot_single_scatter


def save_fig_as_png(fig, filepath, width=1200, height=800, scale=1, engine="kaleido"):
    """Make sure file extension is ".png"
    """
    if os.path.sep in filepath:
        os.makedirs(os.path.sep.join(str(filepath).split(os.path.sep )[:-1]), exist_ok=True)
    fig.write_image(filepath, width=width, height=height, scale=scale, engine=engine)


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