import plotly.express as px
import plotly.graph_objects as go

### Bubble chart that occupies half of a PowerPoint slide
fig = px.scatter(
    data_to_plot,
    x="x_col",
    y="y_col",
    color="color_col",
    size="size_col",
    hover_name="color_col",
    size_max=60,
    text="color_col",
#     trendline="lowess", trendline_color_override="orange"
#     trendline="rolling", trendline_options=dict(window=5)
    labels=dict(
        x_col="x_col_name",
        y_col="y_col_name"
    )
)
fig.update_layout(
    width=750,
    height=750,
    showlegend=False,
    margin=dict(r=30, b=30),
#     xaxis_range=[0, 100],
#     yaxis_range=[0, 0.6],
#     yaxis_tickformat=".0%",
    title=dict(text="plot_title", font=dict(size=26)),
)
fig.show()


### Horizontal bar chart with bars split by colors
fig = px.bar(
    data_to_plot,
    x="x_col",
    y="y_col",
    color="color_col",
    text="text_col",
    orientation="h",
    color_discrete_sequence=px.colors.qualitative.D3,  # Better orange/blue if only two colors, optionally provide a list of colors
    labels=dict(
        y_col="y_col_name",
        x_col="x_col_name"
    )
)
# Sizing to fit inside of a PowerPoint slide
fig.update_layout(
    width=1200,
    height=550,
    margin=dict(r=30, b=30),
#     xaxis_tickformat=".0%",
    title=dict(text="plot_title", font=dict(size=26)),
)
fig.update_traces(texttemplate="%{text:.2s}")  # Formatting the text in the graph
fig.show()

### Sorting the above by overall size while maintaining colors
data_to_plot = df[['y_col', 'x_col_1', 'x_col_2']].copy()
data_to_plot = data_to_plot.melt(id_vars='y_col').rename(columns={'variable': 'x_col_2', 'value': 'x_col_1'})

# Adding a custom column for sorting
sort_mapping = df[['y_col', 'x_col_1']].sort_values('x_col_1', ascending=False)
sort_mapping['sort_order'] = range(0, len(sort_mapping))
sort_mapping = sort_mapping.drop(columns=['x_col_1'])

# Joining the sort mapping and sorting
data_to_plot = data_to_plot.merge(sort_mapping)
data_to_plot = data_to_plot.sort_values('sort_order', ascending=False)


### Adding an annotation w/ out an arrow
fig.add_annotation(
    x=100,
    y=100,
    text="Replace the annotation<br>text here",
    showarrow=False,
    font_size=16,
    opacity=0.85
)
