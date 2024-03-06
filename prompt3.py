import os
from constants import openai_key
from langchain_community.llms import OpenAI
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts.few_shot import FewShotPromptTemplate

os.environ["OPENAI_API_KEY"]=openai_key

st.title("Langchain Prompt with OpenAI")

llm = OpenAI(temperature=0.8)

examples=[{"word":"happy","antonym":"sad"},{"word":"laugh","antonym":"cry"}]

input_prompt=PromptTemplate(
    input_variables=['word','antonym'],
    template="word:{word}\nantonym:{antonym}"
)

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=input_prompt,
    prefix="Give the antonym of every word",
    suffix="Word:{input}\nAntonym: ",
    input_variables=["input"],
)
chain=LLMChain(llm=llm,prompt=prompt)
print(prompt.format(input='big'))
st.write(chain.invoke({"input":"big"}))