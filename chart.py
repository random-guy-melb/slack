from datetime import datetime, timedelta
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from typing import List, Dict, Optional, Literal
from enum import Enum


class ChartType(str, Enum):
    PIE = "pie"
    BAR = "bar"
    DOT = "dot"


class ChartConfig:
    def __init__(
            self,
            field_name: str,
            display_name: str,
            colors: List[str],
            min_percentage: float = 1.5,
            split_values: bool = False,
            default_value: str = "Unknown",
            sort_values: bool = True,
            max_items: int = 15  # Limit number of items in bar/dot plots
    ):
        """
        Configuration for a single chart

        Parameters:
        - field_name: The column name in the DataFrame
        - display_name: The name to display in the chart title
        - colors: List of colors to use in the chart
        - min_percentage: Minimum percentage for a value to be shown separately (pie charts)
        - split_values: Whether to split comma-separated values
        - default_value: Default value for missing data
        - sort_values: Whether to sort values by frequency
        - max_items: Maximum number of items to show in bar/dot plots
        """
        self.field_name = field_name
        self.display_name = display_name
        self.colors = colors
        self.min_percentage = min_percentage
        self.split_values = split_values
        self.default_value = default_value
        self.sort_values = sort_values
        self.max_items = max_items


class DashboardConfig:
    def __init__(
            self,
            date_field: str = "Date",
            charts: List[ChartConfig] = None,
            date_format: Optional[str] = None,
            default_days: int = 7,
            chart_type: ChartType = ChartType.PIE
    ):
        """
        Configuration for the entire dashboard

        Parameters:
        - date_field: Name of the date column
        - charts: List of chart configurations
        - date_format: Format string for parsing dates
        - default_days: Default number of days to show
        - chart_type: Type of chart to display (pie, bar, or dot)
        """
        self.date_field = date_field
        self.charts = charts or []
        self.date_format = date_format
        self.default_days = default_days
        self.chart_type = chart_type


def aggregate_values(series: pd.Series, min_percentage: float = 1.5) -> pd.Series:
    """Aggregate small values into a 'Remaining' category"""
    numeric_series = pd.to_numeric(series.map(lambda x: 1 if isinstance(x, str) else x),
                                   errors='coerce').fillna(1)

    total = numeric_series.sum()
    if total == 0:
        return series

    mask = (numeric_series / total * 100) >= min_percentage
    main_values = series[mask]
    small_values_sum = numeric_series[~mask].sum()

    if small_values_sum > 0:
        other_series = pd.Series({'Remaining': small_values_sum})
        main_values = pd.concat([main_values, other_series])

    return main_values


def create_pie_figure(labels, values, title: str, colors=None, legend_font_size: int = 12):
    """Create a pie chart figure"""
    wrapped_labels = [label[:16] + '<br>' + label[16:] if len(label) > 16 else label
                      for label in labels]

    if len(labels) == 0:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[dict(text="No data available", showarrow=False, font_size=12)],
            width=500
        )
        return fig

    fig = go.Figure()
    fig.add_trace(
        go.Pie(
            labels=wrapped_labels,
            values=values,
            name=title,
            textinfo='percent',
            hoverinfo='label+percent',
            marker_colors=colors,
            showlegend=True,
            textposition='inside'
        )
    )

    fig.update_layout(
        title=title,
        showlegend=True,
        height=450,
        width=500,
        legend=dict(font=dict(size=legend_font_size)),
        uniformtext=dict(mode='hide', minsize=12),
    )
    return fig


def create_bar_figure(labels, values, title: str, colors=None, legend_font_size: int = 12):
    """Create a horizontal bar chart figure"""
    if len(labels) == 0:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[dict(text="No data available", showarrow=False, font_size=12)],
            width=500
        )
        return fig

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels,
            x=values,
            orientation='h',
            marker_color=colors[0] if colors else None,
            text=values,
            textposition='auto',
        )
    )

    fig.update_layout(
        title=title,
        height=max(450, len(labels) * 25),  # Adjust height based on number of bars
        width=500,
        xaxis_title="Count",
        yaxis=dict(
            title="",
            autorange="reversed"  # Display bars from top to bottom
        ),
        showlegend=False,
        margin=dict(l=200)  # Add more margin for labels
    )
    return fig


def create_dot_figure(labels, values, title: str, colors=None, legend_font_size: int = 12):
    """Create a dot plot figure"""
    if len(labels) == 0:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[dict(text="No data available", showarrow=False, font_size=12)],
            width=500
        )
        return fig

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=labels,
            x=values,
            mode='markers+text',
            marker=dict(
                size=12,
                color=colors[0] if colors else None,
            ),
            text=values,
            textposition='middle right',
            textfont=dict(size=10),
        )
    )

    fig.update_layout(
        title=title,
        height=max(450, len(labels) * 25),
        width=500,
        xaxis_title="Count",
        yaxis=dict(
            title="",
            autorange="reversed"
        ),
        showlegend=False,
        margin=dict(l=200)
    )
    return fig


def get_field_values(df: pd.DataFrame, field: str, split_values: bool = False) -> List[str]:
    """Extract unique values from a field, optionally splitting comma-separated values"""
    if not split_values:
        return sorted(df[field].dropna().unique())

    all_values = []
    for values in df[field].dropna():
        if isinstance(values, str):
            values_list = [v.strip() for v in values.split(',')]
            all_values.extend(values_list)
    return sorted(set(all_values))


def aggregate_field_values(
        df: pd.DataFrame,
        field: str,
        split_values: bool = False,
        min_percentage: float = 1.5,
        default_value: str = "Unknown",
        sort_values: bool = True,
        max_items: int = 15,
        chart_type: ChartType = ChartType.PIE
) -> pd.Series:
    """Aggregate values for a field, handling split values if needed"""
    if not split_values:
        series = df[field].apply(
            lambda x: default_value if pd.isna(x) or not str(x).strip() else x
        ).value_counts()
    else:
        all_values = []
        for values in df[field].dropna():
            if isinstance(values, str):
                values_list = [v.strip() for v in values.split(',')]
                all_values.extend(values_list)

        if not all_values:
            return pd.Series()

        series = pd.Series(all_values).value_counts()

    # For pie charts, aggregate small values
    if chart_type == ChartType.PIE:
        return aggregate_values(series, min_percentage)

    # For bar and dot plots, limit number of items and optionally sort
    if sort_values:
        series = series.sort_values(ascending=False)

    return series.head(max_items)


def create_figure(
        labels,
        values,
        title: str,
        colors=None,
        chart_type: ChartType = ChartType.PIE
):
    """Create a figure based on the specified chart type"""
    if chart_type == ChartType.PIE:
        return create_pie_figure(labels, values, title, colors)
    elif chart_type == ChartType.BAR:
        return create_bar_figure(labels, values, title, colors)
    elif chart_type == ChartType.DOT:
        return create_dot_figure(labels, values, title, colors)
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")


def create_dashboard(
        data: Dict,
        config: DashboardConfig,
        unique_id: str
):
    """
    Create a configurable dashboard

    Parameters:
    - data: Dictionary containing the data
    - config: Dashboard configuration
    - unique_id: Unique identifier for widget keys
    """
    # Convert data to DataFrame and process dates
    df = pd.DataFrame(data)
    if df[config.date_field].dtype == "object":
        try:
            if config.date_format:
                df[config.date_field] = pd.to_datetime(df[config.date_field],
                                                       format=config.date_format)
            else:
                df[config.date_field] = pd.to_datetime(df[config.date_field])
        except ValueError as e:
            st.error(f"Error parsing dates: {str(e)}")
            return

    # Set up date range
    max_date = df[config.date_field].max()
    min_date = df[config.date_field].min()
    default_start = max(min_date, max_date - timedelta(days=config.default_days))

    try:
        # Create filters
        filter_cols = st.columns([1] * (len(config.charts) + 2))  # +2 for date and reset

        # Create field filters
        filters = {}
        for i, chart_config in enumerate(config.charts):
            with filter_cols[i]:
                field_values = get_field_values(df, chart_config.field_name,
                                                chart_config.split_values)
                filters[chart_config.field_name] = st.multiselect(
                    f"**{chart_config.display_name}**",
                    options=field_values,
                    key=f"{chart_config.field_name}_filter_{unique_id}"
                )

        # Date filter
        with filter_cols[-2]:
            default_dates = (default_start.date(), max_date.date())
            date_range = st.date_input(
                "**Date Range**",
                value=default_dates,
                min_value=min_date.date(),
                max_value=max_date.date(),
                key=f"date_filter_{unique_id}"
            )

        # Reset button
        with filter_cols[-1]:
            st.write("")  # Add spacing
            st.write("")  # Add spacing
            if st.button("**Reset Dates**", key=f"reset_button_{unique_id}"):
                date_range = default_dates

        # Filter DataFrame
        filtered_df = df.copy()
        for field, values in filters.items():
            if values:
                chart_config = next(c for c in config.charts if c.field_name == field)
                if chart_config.split_values:
                    mask = filtered_df[field].apply(
                        lambda x: any(val in str(x) for val in values) if pd.notna(x) else False
                    )
                else:
                    mask = filtered_df[field].isin(values)
                filtered_df = filtered_df[mask]

        if len(date_range) == 2:
            start_date = datetime.combine(date_range[0], datetime.min.time())
            end_date = datetime.combine(date_range[1], datetime.max.time())
            filtered_df = filtered_df[
                (filtered_df[config.date_field] >= start_date) &
                (filtered_df[config.date_field] <= end_date)
                ]

        # Display record count
        st.metric("Record Count", len(filtered_df))

        # Create charts
        num_charts = len(config.charts)
        chart_cols = st.columns(min(num_charts, 3))

        for i, chart_config in enumerate(config.charts):
            col_idx = i % 3
            with chart_cols[col_idx]:
                summary = aggregate_field_values(
                    filtered_df,
                    chart_config.field_name,
                    chart_config.split_values,
                    chart_config.min_percentage,
                    chart_config.default_value,
                    chart_config.sort_values,
                    chart_config.max_items,
                    config.chart_type
                )

                fig = create_figure(
                    summary.index,
                    summary.values,
                    chart_config.display_name,
                    colors=chart_config.colors[:len(summary)],
                    chart_type=config.chart_type
                )
                st.plotly_chart(fig, use_container_width=True,
                                key=f"plot_{unique_id}_{i}")

        # Display scrollable dataframe
        st.markdown("### Summary")
        st.dataframe(
            filtered_df.sort_values(config.date_field, ascending=False),
            use_container_width=True,
            height=200
        )

    except Exception as e:
        st.error(f"An error occurred while creating the dashboard: {str(e)}")


# Example usage:
if __name__ == "__main__":
    # Define color schemes
    BLUE_COLORS = ['#1E90FF', '#87CEEB', '#4682B4', '#6495ED', '#87CEFA']
    PURPLE_COLORS = ['#9370DB', '#8A2BE2', '#9932CC', '#BA55D3', '#DDA0DD']
    GREEN_COLORS = ['#98FB98', '#90EE90', '#32CD32', '#3CB371', '#2E8B57']

    # Create chart configurations
    charts = [
        ChartConfig(
            field_name="High Level Cause",
            display_name="High Level Cause",
            colors=BLUE_COLORS,
            sort_values=True,
            max_items=10
        ),
        ChartConfig(
            field_name="Tag",
            display_name="Tag Type",
            colors=PURPLE_COLORS,
            default_value="Untagged",
            sort_values=True,
            max_items=10
        ),
        ChartConfig(
            field_name="Assignment Group",
            display_name="Assignment Group",
            colors=GREEN_COLORS,
            split_values=True,
            sort_values=True,
            max_items=12
        )
    ]

    # Sample data
    sample_data = {
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        "High Level Cause": ["Cause A", "Cause B", "Cause A", "Cause C"],
        "Assignment Group": ["@sre-commerce, @sre-api", "@sre-commerce", "@sre-data", "@sre-platform"],
        "Tag": ["Tag1", "Tag2", "Tag1", "Tag3"]
    }

    # Example 1: Pie Chart Dashboard
    pie_config = DashboardConfig(
        date_field="Date",
        charts=charts,
        default_days=7,
        chart_type=ChartType.PIE
    )
    st.subheader("Pie Chart Dashboard")
    create_dashboard(sample_data, pie_config, "dashboard_pie")

    # Example 2: Bar Chart Dashboard
    bar_config = DashboardConfig(
        date_field="Date",
        charts=charts,
        default_days=7,
        chart_type=ChartType.BAR
    )
    st.subheader("Bar Chart Dashboard")
    create_dashboard(sample_data, bar_config, "dashboard_bar")

    # Example 3: Dot Plot Dashboard
    dot_config = DashboardConfig(
        date_field="Date",
        charts=charts,
        default_days=7,
        chart_type=ChartType.DOT
    )
    st.subheader("Dot Plot Dashboard")
    create_dashboard(sample_data, dot_config, "dashboard_dot")
