# app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson

st.set_page_config(layout="wide")

# --- Sidebar Controls ---
st.sidebar.header("Controls")
grid_size = st.sidebar.slider("Grid Size", min_value=2, max_value=20, value=5, step=1)
num_points = st.sidebar.slider("Number of Points", min_value=10, max_value=5000, value=100, step=10)
fit_normal = st.sidebar.checkbox("Fit Normal Distribution")
fit_poisson = st.sidebar.checkbox("Fit Poisson Distribution")

# --- Generate Random Points ---
x_points = np.random.uniform(0, grid_size, num_points)
y_points = np.random.uniform(0, grid_size, num_points)
x_bins = np.floor(x_points).astype(int)
y_bins = np.floor(y_points).astype(int)

# --- Count Points Per Square ---
counts = np.zeros((grid_size, grid_size), dtype=int)
for x, y in zip(x_bins, y_bins):
    if 0 <= x < grid_size and 0 <= y < grid_size:
        counts[y, x] += 1
square_counts = counts.flatten()

# --- Layout ---
col1, col2, col3 = st.columns([3, 2, 1])

# --- Column 1: Grid Plot ---
with col1:
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(grid_size + 1):
        ax.axhline(i, color='gray', lw=0.8)
        ax.axvline(i, color='gray', lw=0.8)

    ax.scatter(x_points, y_points, s=10, color='blue', alpha=0.6)
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{grid_size}×{grid_size} Grid with {num_points} Points', fontsize=12)

    for row in range(grid_size):
        for col in range(grid_size):
            count = counts[row, col]
            ax.text(col + 0.5, row + 0.5, str(count), color='black',
                    ha='center', va='center', fontsize=9)

    st.pyplot(fig)

# --- Column 2: Histogram ---
with col2:
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    bins = np.arange(square_counts.max() + 2) - 0.5
    values, _, _ = ax2.hist(square_counts, bins=bins, color='blue', alpha=0.6,
                            edgecolor='black', label='Observed')

    x_vals = np.arange(0, square_counts.max() + 1)
    if fit_normal:
        mu, sigma = np.mean(square_counts), np.std(square_counts)
        y_vals = norm.pdf(x_vals, mu, sigma) * len(square_counts)
        ax2.plot(x_vals, y_vals, 'r-', lw=2, label='Normal Fit')

    if fit_poisson:
        lam = np.mean(square_counts)
        y_vals = poisson.pmf(x_vals, lam) * len(square_counts)
        ax2.plot(x_vals, y_vals, 'g--', lw=2, label='Poisson Fit')

    ax2.set_xlabel('Points per Square', fontsize=10)
    ax2.set_ylabel('Number of Squares', fontsize=10)
    ax2.set_title('Distribution of Points per Square', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.3)

    st.pyplot(fig2)

# --- Column 3: Stats ---
with col3:
    st.markdown("### Descriptive Stats")
    mean = np.mean(square_counts)
    median = np.median(square_counts)
    std = np.std(square_counts)
    q1 = np.percentile(square_counts, 25)
    q3 = np.percentile(square_counts, 75)
    iqr = q3 - q1
    outliers = np.sum((square_counts < q1 - 1.5 * iqr) | (square_counts > q3 + 1.5 * iqr))

    st.markdown(f"- **Mean:** {mean:.2f}")
    st.markdown(f"- **Median:** {median:.2f}")
    st.markdown(f"- **Std Dev:** {std:.2f}")
    st.markdown(f"- **IQR:** {iqr:.2f}")
    st.markdown(f"- **Outliers:** {outliers}")