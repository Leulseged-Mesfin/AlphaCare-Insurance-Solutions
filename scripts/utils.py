import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def fill_null_values(df):
    for column in df.columns:
        if df[column].dtype == 'object' and df[column].dtype == 'category':
            # Fill missing values with the previous value (forward fill)
            df[column].fillna(method='ffill', inplace=True)
        elif df[column].dtype == 'float64' and df[column].dtype == 'int64':
            # Fill missing values with 0
            df[column].fillna(0, inplace=True)
    return df


# Univariate Analysis
def plot_numerical_columns(df):
    
    # Plotting histograms for numerical columns
    numerical_columns = ['CalculatedPremiumPerTerm','TotalPremium', 'TotalClaims']

    for column in numerical_columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[column], kde=True, bins=10)  # kde=True adds a smooth density curve
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()



def plot_category_columns(df):
    # Plotting bar charts for categorical columns
    categorical_columns = [ 'Citizenship', 'Title', 'AccountType', 'Gender', 'Country', 'make', 'CoverType']

    for column in categorical_columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=column, data=df)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.show()


def scater_plot_for_TP_TC(df):
    # Relationship between TotalPremium and TotalClaims across PostalCodes
    # plt.figure(figsize=(8, 6))
    plt.figure(figsize=(10, 6), dpi=80)  # Lower the DPI if needed

    sns.scatterplot(x='TotalPremium', y='TotalClaims', data=df, palette='Set2')
    plt.title('Scatter Plot of TotalPremium vs TotalClaims by PostalCode')
    plt.xlabel('Total Premium')
    plt.ylabel('Total Claims')
    plt.show()