import os
from constants import openai_key
from langchain_community.llms import OpenAI
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

os.environ["OPENAI_API_KEY"]=openai_key

st.title("Langchain Prompt with OpenAI")

input_text=st.text_input("Search here")

llm = OpenAI(temperature=0.8)

input_prompt=PromptTemplate(
    input_variables=['name'],
    template="Tell me about {name}"
)

chain=LLMChain(llm=llm,prompt=input_prompt)
if(input_text):
    st.write(chain.invoke({"name":input_text}))