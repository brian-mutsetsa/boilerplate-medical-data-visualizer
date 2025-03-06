import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1) Import the data from medical_examination.csv and assign it to the df variable.
df = pd.read_csv("medical_examination.csv")

# 2) Add an overweight column to the data.
# Calculate BMI (weight in kg / (height in m)^2). If BMI > 25, overweight = 1; otherwise 0.
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3) Normalize data by making 0 always good and 1 always bad.
# For cholesterol and gluc: 1 -> 0 and >1 -> 1.
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

def draw_cat_plot():
    # 5) Create a DataFrame for the cat plot using pd.melt with the specified columns.
    df_cat = pd.melt(df, id_vars=["cardio"],
                     value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"])
    
    # 6) Group and reformat the data: count each combination of 'cardio', 'variable', and 'value'.
    df_cat = df_cat.groupby(["cardio", "variable", "value"]).size().reset_index(name="total")
    
    # 7) (Already converted into long format via pd.melt and grouped above.)
    
    # 8) Draw the categorical plot using seaborn's catplot and store the figure.
    fig = sns.catplot(
        data=df_cat, kind="bar",
        x="variable", y="total", hue="value", col="cardio"
    ).fig
    
    # 9) Save the figure and return it.
    fig.savefig('catplot.png')
    return fig

def draw_heat_map():
    
    # 11) Clean the data by filtering out incorrect patient data.
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]
    
    # 12) Calculate the correlation matrix.
    corr = df_heat.corr()
    
    # 13) Generate a mask for the upper triangle of the correlation matrix.
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # 14) Set up the matplotlib figure.
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 15) Draw the heatmap using seaborn's heatmap function.
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", square=True,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    
    # 16) Save the figure and return it.
    fig.savefig('heatmap.png')
    return fig