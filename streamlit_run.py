import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

# Initialize the LLM
llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", temperature=0.1,
                        max_new_tokens=80, streaming=True, top_k=80, top_p=0.95)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert Python Pandas Query Generator. Your task is to convert natural language questions into precise, executable Pandas queries.

Instructions:
1. Use only the provided column names in your queries.
2. Always use 'df' as the DataFrame name.
3. Generate only the Pandas query, without any explanations or additional text.
4. If the question cannot be answered with the given columns, respond with exactly: 'I cannot answer this question'
5. Ensure your queries are syntactically correct and follow Pandas best practices.
6. Use appropriate Pandas functions and methods (e.g., mean(), median(), mode(), describe(), etc.) when applicable.
7. For comparisons, use proper syntax (e.g., df['column'] == value).
8. For string operations, use str accessor (e.g., df['column'].str.contains()).
9. Handle potential NaN values when necessary.
10. For multiple statistics, use a single line of code when possible.

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

Remember to return ONLY the Pandas query or the exact 'I cannot answer this question' response."""),
    ("human", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

def clean_string(text):
    cleaned_text = text.replace("Python:", "").replace("</s>", "").replace("Output: ", "").replace("I: ", "")
    cleaned_text = cleaned_text.strip()
    return cleaned_text

st.title("Pandas Query Generator")

# File uploader for CSV files
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("DataFrame Preview:")
    st.dataframe(df.head())

    # Input for the user's question
    user_input = st.text_input("Enter your question:")

    if st.button("Generate Query"):
        if user_input:
            # Generate the query using the LLM
            columns = list(df.columns)
            query = f"Input: question: '{user_input}', columns: {columns}"
            response = chain.invoke({"input": query})

            # Clean the generated query
            code_string = clean_string(response)

            st.subheader("Generated Pandas Query:")
            st.code(code_string, language='python')

            # Prepare the execution context
            exec_context = {"df": df}

            try:
                # Execute the generated query
                result = eval(code_string, exec_context)
                st.subheader("Query Result:")
                st.write(result)
            except Exception as e:
                st.error(f"Couldn't execute the query: {str(e)}")
                st.text("Generated query:")
                st.code(code_string, language='python')
        else:
            st.warning("Please enter a question.")
else:
    st.info("Please upload a CSV file to get started.")