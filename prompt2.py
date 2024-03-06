import os
from constants import openai_key
from langchain_community.llms import OpenAI
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts.few_shot import FewShotPromptTemplate

os.environ["OPENAI_API_KEY"]=openai_key

st.title("Langchain Prompt with OpenAI")

# input_text=st.text_input("Search here")

llm = OpenAI(temperature=0.8)

input_prompt=PromptTemplate(
    input_variables=['sentence','targeted_speech'],
    template="In as easy way translate the following sentence from {sentence} to {targeted_speech}"
)

input_prompt.format(sentence="How are you?",targeted_speech="hindi")

chain=LLMChain(llm=llm,prompt=input_prompt)

st.write(chain.invoke({"sentence":"Hi how are you","targeted_speech":"german"}))