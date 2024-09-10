import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import matplotlib.pyplot as plt
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import os

# Load the CSV file
df = pd.read_csv("data_science_internship_test.csv")
# Create a Pandas DataFrame agent
agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0),
    df,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    allow_dangerous_code=True
)
def query_data(query):
    result = agent.run(query)
    return result
def plot_data(data, x_column, y_column, plot_type='line'):
    plt.figure(figsize=(10, 6))
    if plot_type == 'line':
        plt.plot(data[x_column], data[y_column])
    elif plot_type == 'bar':
        plt.bar(data[x_column], data[y_column])
    elif plot_type == 'scatter':
        plt.scatter(data[x_column], data[y_column])
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f"{y_column} vs {x_column}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
# Main loop
while True:
    user_query = input("Enter your query (or 'quit' to exit): ")
    if user_query.lower() == 'quit':
        break
    result = query_data(user_query)
    print(result)
    if isinstance(result, pd.DataFrame) and len(result) > 1:
        plot_choice = input("Do you want to plot this data? (yes/no): ")
        if plot_choice.lower() == 'yes':
            x_column = input("Enter the column name for the x-axis: ")
            y_column = input("Enter the column name for the y-axis: ")
            plot_type = input("Enter the plot type (line/bar/scatter): ")
            plot_data(result, x_column, y_column, plot_type)