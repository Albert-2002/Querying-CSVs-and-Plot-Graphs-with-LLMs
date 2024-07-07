import streamlit as st
import pandas as pd
import seaborn as sns
import os
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import matplotlib.pyplot as plt

# meta-llama/Meta-Llama-3-8B-Instruct
# mistralai/Mistral-7B-Instruct-v0.2
# mistralai/Mistral-7B-Instruct-v0.3

load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", temperature=0.1,
                        max_new_tokens=150, streaming=True, top_k=80, top_p=0.95)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert Python Pandas Query and Matplotlib Chart Generator. Convert natural language questions into executable Pandas queries or Matplotlib chart generation code.

Rules:
1. Use only provided column names.
2. Always use 'df' as the DataFrame name.
3. Generate only Pandas query or Matplotlib code.
4. If question can't be answered with given columns, return: 'I cannot answer this question'
5. Ensure code is syntactically correct and follows best practices.
6. Use appropriate Pandas functions (e.g., mean(), median(), mode(), describe()).
7. Use proper syntax for comparisons (e.g., df['column'] == value).
8. Use str accessor for string operations (e.g., df['column'].str.contains()).
9. Handle NaN values when necessary.
10. Use single line of code for multiple statistics when possible.
11. For charts, use appropriate Matplotlib functions with proper labels and styling.

Examples:

Input: question: 'What is the average age?', columns: ['Age', 'Gender', 'Height']
Output: df['Age'].mean()

Input: question: 'How many females are taller than 160cm?', columns: ['Age', 'Gender', 'Height']
Output: df[(df['Gender'] == 'Female') & (df['Height'] > 160)].shape[0]

Input: question: 'What is the oldest age for each gender?', columns: ['Age', 'Gender', 'Height']
Output: df.groupby('Gender')['Age'].max()

Input: question: 'Get mean, median and mode of the Age column', columns: ['Age', 'Gender', 'Height']
Output: df['Age'].agg(['mean', 'median', lambda x: x.mode().iloc[0]])

Input: question: 'What is the average height of males over 30?', columns: ['Age', 'Gender', 'Height']
Output: df[(df['Gender'] == 'Male') & (df['Age'] > 30)]['Height'].mean()

Input: question: 'Get mean, median and mode of the Glucose column', columns: ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
Output: df['Glucose'].agg(['mean', 'median', lambda x: x.mode().iloc[0]])

Input: question: 'Get the minimum and maximum of all columns', columns: ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
Output: df.agg(['min', 'max'])

Input: question: 'CHART: Show the distribution of ages', columns: ['Age', 'Gender', 'Height']
Output: df['Age'].plot(kind='hist', title='Distribution of Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

Do not include any explanations or additional text in your response.
Convert questions to Pandas/Matplotlib code. Use only given columns and 'df' as DataFrame name. Return only code, no explanations.
If impossible, return 'I cannot answer this question'."""),
    ("human", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

def clean_string(text):
    cleaned_text = text.replace("Python:", "").replace("</s>", "").replace("Output: ", "").replace("I: ", "")
    cleaned_text = cleaned_text.strip()
    return cleaned_text

st.set_page_config(page_title="CSV Analyzer📊", page_icon="📈")
st.title("Pandas Query📰 and Chart Generator📊")

uploaded_file = st.file_uploader("Upload your CSV file:", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("DataFrame Preview: ")
    st.dataframe(df.head())

    user_input = st.text_input("Enter your prompt: ",placeholder="Start prompt with 'CHART:' to enter visualization mode.")

    if st.button("Generate Query or Chart 👈"):
        if user_input:
            columns = list(df.columns)
            query = f"Input: question: '{user_input}', columns: {columns}"
            response = chain.invoke({"input": query})

            code_string = clean_string(response)

            st.subheader("Generated Code:")
            st.code(code_string, language='python')

            exec_context = {"df": df, "plt": plt, "sns": sns}

            try:
                if user_input.startswith("CHART:"):
                    exec(code_string, exec_context)
                    st.pyplot(plt)
                else:
                    result = eval(code_string, exec_context)
                    st.subheader("Query Result:")
                    st.write(result)
            except Exception as e:
                st.error(f"Couldn't execute the code: {str(e)}")
                st.text("Generated code:")
                st.code(code_string, language='python')
        else:
            st.warning("Please enter a question.")
else:
    st.info("Please upload a CSV file to get started.")