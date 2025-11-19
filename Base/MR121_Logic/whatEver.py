import plotly.express as px
import pandas as pd

# Sample data
df = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [10, 11, 12, 13, 14],
    'z': [5, 6, 7, 8, 9],
    'color': ['A', 'B', 'A', 'B', 'A']
})

# Create 3D scatter plot
fig = px.scatter_3d(df, x='x', y='y', z='z', color='color', size_max=10)
fig.show()