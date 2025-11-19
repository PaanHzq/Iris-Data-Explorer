import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import altair as alt

st.set_page_config(
    page_title="Iris Data Explorer",
    layout="wide",
)

@st.cache_data
def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame

    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    return df, iris

df, iris = load_data()


st.title("Iris Data Explorer")
st.markdown(
    """
This interactive web app lets you explore the classic **Iris flower dataset**.  
Use the sidebar filters to subset the data and view different charts.
"""
)


st.sidebar.header("Filters")


species_options = ["All"] + list(iris.target_names)
selected_species = st.sidebar.multiselect(
    "Select species",
    options=species_options,
    default=["All"],
)


filtered_df = df.copy()


if "All" not in selected_species:
    filtered_df = filtered_df[filtered_df["species"].isin(selected_species)]


feature_to_filter = st.sidebar.selectbox(
    "Filter by feature",
    iris.feature_names
)

min_val = float(filtered_df[feature_to_filter].min())
max_val = float(filtered_df[feature_to_filter].max())

value_range = st.sidebar.slider(
    "Value range",
    min_value=round(min_val, 1),
    max_value=round(max_val, 1),
    value=(round(min_val, 1), round(max_val, 1)),
)

filtered_df = filtered_df[
    (filtered_df[feature_to_filter] >= value_range[0]) &
    (filtered_df[feature_to_filter] <= value_range[1])
]


st.subheader("Data Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Number of rows", len(filtered_df))
col2.metric("Number of columns", filtered_df.shape[1])
col3.metric("Unique species", filtered_df["species"].nunique())

st.markdown("### Data Preview")
st.dataframe(filtered_df, use_container_width=True)


st.markdown("## Visualizations")

viz_type = st.radio(
    "Choose visualization type:",
    ["Scatter Plot", "Histogram"],
    horizontal=True,
)

numeric_features = iris.feature_names

if viz_type == "Scatter Plot":
    st.markdown("### Scatter Plot")

    col_x, col_y = st.columns(2)
    x_axis = col_x.selectbox("X-axis", numeric_features, index=0)
    y_axis = col_y.selectbox("Y-axis", numeric_features, index=1)

    scatter_chart = (
        alt.Chart(filtered_df)
        .mark_circle(size=70)
        .encode(
            x=alt.X(x_axis, title=x_axis),
            y=alt.Y(y_axis, title=y_axis),
            color=alt.Color("species", title="Species"),
            tooltip=numeric_features + ["species"],
        )
        .interactive()
    )

    st.altair_chart(scatter_chart, use_container_width=True)

else:
    st.markdown("### Histogram")

    feature_hist = st.selectbox("Feature", numeric_features)
    bins = st.slider("Number of bins", min_value=5, max_value=50, value=20)

    hist_chart = (
        alt.Chart(filtered_df)
        .mark_bar()
        .encode(
            x=alt.X(feature_hist, bin=alt.Bin(maxbins=bins), title=feature_hist),
            y=alt.Y("count()", title="Count"),
            color=alt.Color("species", title="Species"),
            tooltip=["species", feature_hist],
        )
    )

    st.altair_chart(hist_chart, use_container_width=True)

st.markdown("---")

