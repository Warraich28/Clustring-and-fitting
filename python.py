# Import numpy for numerical operations and linear algebra
import numpy as np

# Import pandas for data manipulation and analysis
import pandas as pd

# Import scipy.optimize for optimization algorithms and tools
import scipy.optimize as opt

# Import KMeans clustering algorithm from scikit-learn library
from sklearn.cluster import KMeans

# Import make_blobs for generating sample data with blobs
from sklearn.datasets import make_blobs

# Import matplotlib for data visualization and plotting
import matplotlib.pyplot as plt

import itertools as iter


cluster_count = 4

def generate_data(num_samples: int, num_clusters: int):
    """
    Generate sample data with specified number of samples and clusters
    :param num_samples: number of samples to generate
    :param num_clusters: number of clusters in the generated data
    :return: generated data (X, y)
    """

    X, y = make_blobs(n_samples=num_samples, centers=num_clusters, random_state=0)
    
    return X, y

X, y = generate_data(200, cluster_count)

def apply_kmeans(data, num_clusters):
    """
    Apply k-means clustering to the provided dataset with the specified number of clusters
    :param data: dataset to apply k-means clustering on
    :param num_clusters: number of clusters for k-means clustering
    :return: k-means model object
    """
    model = KMeans(n_clusters=num_clusters, random_state=0).fit(data)
    return model

kmeans_model = apply_kmeans(X, cluster_count)

def visualize_clusters(data, kmeans_model, x_label, y_label, plot_title):
    """
    Visualize k-means clustering results using a scatter plot
    :param data: dataset with applied k-means clustering
    :param kmeans_model: k-means model object
    :param x_label: label for the x-axis
    :param y_label: label for the y-axis
    :param plot_title: title for the plot
    """
    plt.scatter(data[:, 0], data[:, 1], c=kmeans_model.labels_)

    plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], marker='D', color='black')

    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(plot_title, fontsize=15)
    plt.legend(['Cluster', 'Cluster Center'])
    plt.show()

visualize_clusters(X, kmeans_model, 'X-axis', 'Y-axis', 'K-means Clustering Visualization')


# Analyzing data from different countries
def load_worldbank_data(filepath: str):
    """
    Load a file in Worldbank format and return the original and reshaped dataframes
    :param filepath: path of the file to be loaded
    :return: original dataframe and reshaped dataframe
    """
    # Load the data
    original_data = pd.read_csv(filepath)

    # Extract the country names
    countries = list(original_data['Country Name'])

    # Reshape the dataframe
    reshaped_data = original_data.transpose()

    # Update column names using country names
    reshaped_data.columns = countries

    # Remove unnecessary rows
    reshaped_data = reshaped_data.iloc[4:]
    reshaped_data = reshaped_data.iloc[:-1]

    # Reset dataframe index
    reshaped_data = reshaped_data.reset_index()

    # Rename the index column
    reshaped_data = reshaped_data.rename(columns={"index": "Year"})

    # Convert year column to int
    reshaped_data['Year'] = reshaped_data['Year'].astype(int)

    return original_data, reshaped_data

data1, data2 = load_worldbank_data('data2.csv')



def compute_epc_statistics(data):
    """
    Calculate the average EPC and EPC for a specific year
    :param data: Dataframe containing the EPC information
    :return: None
    """
    # Compute the average EPC (kWh per capita) for each nation
    avg_epc = data.mean()
    print("Average EPC : ")
    print(avg_epc)
    print("\n\n")

    # Compute the EPC (kWh per capita) for each nation in 2013
    epc_year_2013 = data[data['Year'] == 2013]
    print("EPC for 2013 : ")
    print(epc_year_2013)

compute_epc_statistics(data2)



def visualize_epc_trends(data, selected_countries):
    """
    Plot the electric power consumption (EPC) per capita for specified countries over time
    :param data: dataframe with EPC information
    :param selected_countries: list of countries to visualize EPC for
    """
    data.plot(x='Year', y=selected_countries)
    plt.title("Electric power consumption per country", fontsize=15)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("kWh per capita", fontsize=14)
    plt.show()

visualize_epc_trends(data2, ['United States', 'China', 'India'])



def read_worldbank_data(file_path):
    """
    Read Worldbank data file and return original and transposed dataframes.

    :param file_path: str
        Path of the file to be read
    :return: tuple of DataFrames
        Original dataframe and transposed dataframe
    """
    original_df = pd.read_csv(file_path)
    country = list(original_df['Country Name'])

    transposed_df = original_df.transpose()
    transposed_df.columns = country

    transposed_df = transposed_df.iloc[4:]
    transposed_df = transposed_df.iloc[:-1]

    transposed_df = transposed_df.reset_index()
    transposed_df = transposed_df.rename(columns={"index": "Year"})
    transposed_df['Year'] = transposed_df['Year'].astype(int)

    return original_df, transposed_df


original_df, transposed_df = read_worldbank_data("data.csv")

country = list(original_df['Country Name'])


def plot_electric_power_consumption(df, country):
    """
    Plot electric power consumption of a given country over time.

    :param df: DataFrame
        DataFrame containing the electric power consumption data
    :param country: str
        Name of the country for which the electric power consumption should be plotted
    """
    df.plot("Year", country)
    plt.title(f"{country}'s Electric power consumption", fontsize=15)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("kWh per capita", fontsize=14)
    plt.legend(["EPC"])
    plt.show()


plot_electric_power_consumption(transposed_df, country[0])


def logistic(t, n0, g, t0):
    """
    Calculate logistic function with scale factor n0 and growth rate g.

    :param t: array-like
        Input data
    :param n0: float
        Scale factor
    :param g: float
        Growth rate
    :param t0: float
        Inflection point
    :return: array-like
        Logistic function evaluated at t
    """
    f = n0 / (1 + np.exp(-g * (t - t0)))
    return f


df = transposed_df
param, covar = opt.curve_fit(
    logistic, df["Year"], df[country].squeeze(), p0=(float(df[country].iloc[0]), 0.03, 2000.0)
)

sigma = np.sqrt(np.diag(covar))
print("parameters:", param)
print("std. dev.", sigma)

df["fit"] = logistic(df["Year"], *param)

def display_logistic_fit(dataframe, nation):
    """
    Visualize a logistic fit of the specified country's EPC growth
    :param dataframe: DataFrame containing the EPC data
    :param nation: string of the country to plot
    """
    dataframe.plot("Year", [nation, "fit"])
    plt.title("Logistic fit for {}'s EPC growth".format(nation), fontsize=15)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("kWh per capita", fontsize=14)
    plt.legend(["EPC"])
    plt.show()

display_logistic_fit(df, country[0])

upcoming_years = [2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]
# Compute logistic function values for upcoming years
future_log_values = logistic(upcoming_years, *param)

# Display the future logistic values
print("Future Logistic Values from 2021 to 2030:")
print(future_log_values)

years_range = np.arange(df['Year'][0], 2031)
forecast_values = logistic(years_range, *param)

def visualize_future_predictions(dataframe, nation, predictions, years):
    """
    Visualize the future prediction of a country's EPC growth using a logistic model
    :param dataframe: DataFrame containing the EPC data
    :param nation: string of the country to plot
    :param predictions: forecasted values for the future
    :param years: future year
    """
    plt.plot(dataframe["Year"], dataframe[nation], label="EPC")
    plt.plot(years, predictions, label="forecast")

    plt.title("Prediction of {}'s EPC growth using Logistic Model for future years".format(nation), fontsize=15)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("kWh per capita", fontsize=14)
    plt.legend(["EPC"])
    plt.legend()
    plt.show()

visualize_future_predictions(df, country[0], forecast_values, years_range)

df2 = pd.DataFrame({'Upcoming Year': upcoming_years, 'Logistic': future_log_values})
df2

# The err_ranges function remains unchanged as it is a utility function for calculating upper and lower limits.

limits_low, limits_up = err_ranges(years_range, logistic, param, sigma)


def visualize_prediction_intervals(dataframe, nation, predictions, years, lower_bound, upper_bound):
    """
    Visualize the future prediction of a country's EPC growth using a logistic model along with lower and upper limits
    :param dataframe: DataFrame containing the EPC data
    :param nation: string of the country to plot
    :param predictions: forecasted values for the future
    :param years: future year
    :param lower_bound: lower limits for the forecast
    :param upper_bound: upper limits for the forecast
    """
    plt.figure()
    plt.plot(dataframe["Year"], dataframe[nation], label="EPC")
    plt.plot(years, predictions, label="forecast")

    plt.fill_between(years, lower_bound, upper_bound, color="yellow", alpha=0.7)
    plt.title("Lower and Upper Boundaries of {}'s EPC growth".format(nation), fontsize=15)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("kWh per capita", fontsize=14)
    plt.legend()
    plt.show()

visualize_prediction_intervals(df, country[0], forecast, year, low, up)

# The err_ranges function was already modified previously as a utility function for calculating upper and lower limits.
print(err_ranges(2030, logistic, param, sigma))


