import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("medical_examination.csv")
# BMI = weight / height^2 in meters
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)).apply(lambda x: 1 if x > 25 else 0)

df['cholesterol'] = df['cholesterol'].apply(lambda x: 1 if x > 1 else 0)
df['gluc'] = df['gluc'].apply(lambda x: 1 if x > 1 else 0)

def draw_cat_plot():
    df_cat=pd.melt(df,
           id_vars=['cardio'],
           value_vars=['cholesterol','gluc','smoke','alco','active','overweight'])
    df_cat=df_cat.groupby(['cardio','variable','value']).size().reset_index(name='total')

    fig=sns.catplot(data=df_cat,x='variable',y='total',hue='value',col='cardio',kind='bar').fig
    return fig
draw_cat_plot()

def draw_heat_map():
    # Clean data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Correlation matrix
    corr = df_heat.corr()

    # Upper triangle mask
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", center=0, vmax=.3, vmin=-0.1, square=True, linewidths=.5, cbar_kws={"shrink": .5})

    return fig
