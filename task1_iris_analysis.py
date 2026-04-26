"""
Task 1: Exploring and Visualizing the Iris Dataset

Objective:
Understand how to read, summarize, and visualize a dataset using Python.

Dataset:
Iris Dataset — contains measurements of iris flowers:
- sepal_length
- sepal_width
- petal_length
- petal_width
- species

Tools Used:
- pandas
- matplotlib
- seaborn
"""

# ===============================
# Step 1 — Import Libraries
# ===============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ===============================
# Step 2 — Load Dataset
# ===============================

iris = sns.load_dataset("iris")

print("Dataset loaded successfully.\n")


# ===============================
# Step 3 — Dataset Understanding
# ===============================

print("Shape of dataset:")
print(iris.shape)

print("\nColumns:")
print(iris.columns)

print("\nFirst 5 rows:")
print(iris.head())

print("\nDataset Info:")
iris.info()

print("\nStatistical Summary:")
print(iris.describe())


# ===============================
# Step 4 — Data Cleaning
# ===============================

print("\nMissing values check:")
print(iris.isnull().sum())

# No missing values found


# ===============================
# Step 5 — Exploratory Data Analysis (EDA)
# ===============================

# Scatter Plot
plt.figure(figsize=(8, 6))

sns.scatterplot(
    x="sepal_length",
    y="petal_length",
    hue="species",
    data=iris
)

plt.title("Sepal Length vs Petal Length")

plt.savefig("scatter_plot.png")

plt.show()


# Histogram
plt.figure(figsize=(8, 6))

sns.histplot(
    iris["sepal_length"],
    bins=20,
    kde=True
)

plt.title("Distribution of Sepal Length")

plt.savefig("histogram_plot.png")

plt.show()


# Box Plot
plt.figure(figsize=(8, 6))

sns.boxplot(
    x="species",
    y="sepal_length",
    data=iris
)

plt.title("Sepal Length by Species")

plt.savefig("box_plot.png")

plt.show()



"""
Conclusion:

1. Petal length clearly separates species.
2. Setosa species has smaller sepal length.
3. The dataset contains no missing values.
4. No major outliers detected.
"""