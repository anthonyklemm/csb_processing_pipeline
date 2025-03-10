# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:08:15 2024

@author: Anthony.R.Klemm
"""
# After code execution, open up web browser and navigate to this URL:
    # http://127.0.0.1:8050/
import dash
from dash import dcc  
from dash import html
import plotly.express as px
import pandas as pd
#import geopandas as gpd
#import shapely.geometry
#import numpy as np


master_offsets = r"D:\CSB_texas\processed\master_offsets.csv"
leaderboard = r"D:\CSB_texas\processed\processed_exports\leaderboard.csv"

offsets_df = pd.read_csv(master_offsets)
leader_df = pd.read_csv(leaderboard)
offsets_df = offsets_df.query("platform_name != 'Anonymous'")
leader_df = leader_df.query("`Platform Name` != 'Anonymous'")


leader_bar = px.bar(leader_df, x="Platform Name", y="Total Distance (nautical miles)", title="CSB Contributors NE Leaderboard - mileage")
accuracy_score = px.bar(offsets_df, x='platform_name', y='accuracy_score', title="Accuracy score (inverse of std_dev of comparison to reference bathy)")
# After generating the figure, adjust its height
leader_bar.update_layout(height=800)  # Example height in pixels
accuracy_score.update_layout(height=800)  # Same for this figure

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div(children=[

    
    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='leaderboard-contributors',
        figure=leader_bar
    ),
    
    dcc.Graph(
        id='accuracy-score-graph',
        figure=accuracy_score
    ),
    '''

    html.Div([
        html.H1("Tracklines Visualization"),
        dcc.Graph(id='geo-map', figure=fig),
        ])
    '''
])

if __name__ == '__main__':
    app.run_server(debug=True)
