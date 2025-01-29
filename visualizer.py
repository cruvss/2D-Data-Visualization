import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np

# Set the style for all plots - using a built-in style
plt.style.use('fivethirtyeight')  # Alternative styles: 'bmh', 'ggplot', 'classic'

def configure_plot_style(fig, ax):
    """Configure common plot styling elements"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

# Display available styles (optional)
st.sidebar.write("Available Matplotlib Styles:", plt.style.available)

st.title("Interactive Dataset Plotting Tool")

# Upload Dataset
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    try:
        # Load dataset
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df)

        # Plot type selection
        plot_types = ["Line Plot", "Bar Plot", "Scatter Plot", "Histogram", "Box Plot", "Correlation Matrix"]
        plot_type = st.selectbox("Select Plot Type:", plot_types)

        # Color scheme selection
        color_schemes = ['viridis', 'magma', 'plasma', 'inferno', 'cividis']
        color_scheme = st.selectbox("Select Color Scheme:", color_schemes)

        # Common figure creation
        fig, ax = plt.subplots(figsize=(10, 6))
        configure_plot_style(fig, ax)

        if plot_type in ["Line Plot", "Bar Plot"]:
            x_column = st.selectbox("Select X-axis column:", df.columns)
            y_column = st.selectbox("Select Y-axis column:", df.columns)

            if not pd.api.types.is_numeric_dtype(df[y_column]):
                st.warning("Y-axis column must be numeric for this plot type.")
            else:
                if plot_type == "Line Plot":
                    ax.plot(df[x_column], df[y_column], marker='o', linewidth=2, 
                           color=plt.cm.get_cmap(color_scheme)(0.6))
                else:  # Bar Plot
                    bars = ax.bar(df[x_column], df[y_column])
                    for i, bar in enumerate(bars):
                        bar.set_color(plt.cm.get_cmap(color_scheme)(i/len(bars)))
                
                ax.set_title(f"{plot_type} of {y_column} vs {x_column}", pad=20, fontsize=14)
                ax.set_xlabel(x_column, fontsize=12)
                ax.set_ylabel(y_column, fontsize=12)
                plt.xticks(rotation=45 if len(df[x_column].unique()) > 10 else 0)

        elif plot_type == "Scatter Plot":
            x_column = st.selectbox("Select X-axis column:", df.columns)
            y_column = st.selectbox("Select Y-axis column:", df.columns)
            
            if not pd.api.types.is_numeric_dtype(df[x_column]) or not pd.api.types.is_numeric_dtype(df[y_column]):
                st.warning("Both X and Y columns must be numeric for scatter plot.")
            else:
                scatter = ax.scatter(df[x_column], df[y_column], 
                                   c=np.arange(len(df)), cmap=color_scheme, 
                                   alpha=0.6, s=100)
                plt.colorbar(scatter, ax=ax, label='Index')
                ax.set_title(f"Scatter Plot of {y_column} vs {x_column}", pad=20, fontsize=14)
                ax.set_xlabel(x_column, fontsize=12)
                ax.set_ylabel(y_column, fontsize=12)

        elif plot_type == "Histogram":
            column = st.selectbox("Select column:", df.columns)
            bins = st.slider("Number of bins:", min_value=5, max_value=50, value=20)

            if not pd.api.types.is_numeric_dtype(df[column]):
                st.warning("Column must be numeric for histogram.")
            else:
                n, bins, patches = ax.hist(df[column], bins=bins, edgecolor='white', linewidth=1)
                for i, patch in enumerate(patches):
                    patch.set_facecolor(plt.cm.get_cmap(color_scheme)(i/len(patches)))
                
                ax.set_title(f"Histogram of {column}", pad=20, fontsize=14)
                ax.set_xlabel(column, fontsize=12)
                ax.set_ylabel("Frequency", fontsize=12)

        elif plot_type == "Box Plot":
            x_column = st.selectbox("Select grouping column:", df.columns)
            y_column = st.selectbox("Select value column:", df.columns)

            if not pd.api.types.is_numeric_dtype(df[y_column]):
                st.warning("Value column must be numeric for box plot.")
            else:
                box_plot = ax.boxplot([group[1][y_column].values for group in df.groupby(x_column)],
                                    labels=df[x_column].unique(),
                                    patch_artist=True)
                
                # Color the boxes
                colors = [plt.cm.get_cmap(color_scheme)(i/len(box_plot['boxes'])) 
                         for i in range(len(box_plot['boxes']))]
                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax.set_title(f"Box Plot of {y_column} grouped by {x_column}", pad=20, fontsize=14)
                ax.set_xlabel(x_column, fontsize=12)
                ax.set_ylabel(y_column, fontsize=12)
                plt.xticks(rotation=45 if len(df[x_column].unique()) > 10 else 0)

        elif plot_type == "Correlation Matrix":
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            numeric_df = df[numeric_columns]
            
            if len(numeric_columns) == 0:
                st.warning("No numeric columns found in the dataset for correlation matrix.")
            else:
                corr = numeric_df.corr()
                im = ax.imshow(corr, cmap=color_scheme)
                plt.colorbar(im, ax=ax)
                
                # Add correlation values
                for i in range(len(corr)):
                    for j in range(len(corr)):
                        text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                                     ha='center', va='center',
                                     color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')
                
                ax.set_xticks(range(len(corr.columns)))
                ax.set_yticks(range(len(corr.columns)))
                ax.set_xticklabels(corr.columns, rotation=45, ha='right')
                ax.set_yticklabels(corr.columns)
                ax.set_title("Correlation Matrix", pad=20, fontsize=14)

        # Adjust layout and display plot
        plt.tight_layout()
        st.pyplot(fig)
        
        # Download button
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=300, bbox_inches='tight')
        buffer.seek(0)
        st.download_button(
            label="Download Plot as PNG",
            data=buffer,
            file_name="plot.png",
            mime="image/png"
        )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure your dataset is properly formatted and contains appropriate data types for the selected plot type.")