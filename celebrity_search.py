# Integrate our code with OpenAI API
import os
from constants import openai_key
from langchain import PromptTemplate
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import streamlit as st
from langchain.memory import ConversationBufferMemory

os.environ["OPENAI_API_KEY"]=openai_key

# streamlit framework

st.title("Celebrity Search")

input_text=st.text_input("Search about any Celebrity")

# Prompt template

first_input_prompt=PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)

# memory

person_memory=ConversationBufferMemory(input_key='name',memory_key='person_history')
place_memory=ConversationBufferMemory(input_key='person',memory_key='place_history')
state_memory=ConversationBufferMemory(input_key='place',memory_key='state_history')
# openAI LLMS

llm = OpenAI(temperature=0.8)
chain=LLMChain(llm=llm,prompt=first_input_prompt,verbose=True,output_key="person",memory=person_memory)

# Second Prompt template

second_input_prompt=PromptTemplate(
    input_variables=['person'],
    template="What is the city {person} is living in, just give the city name thats it no extra info needed"
)

chain_2=LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key="place",memory=place_memory)

third_input_prompt=PromptTemplate(
    input_variables=['place'],
    template="What is the name of the state in which this {place} is located, just give the state name thats it no extra info needed "
)

chain_3=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key="state",memory=state_memory)

parent_chain=SequentialChain(chains=[chain,chain_2,chain_3],input_variables=['name'],output_variables=['person','place','state'],verbose=True)

if(input_text):{
    st.write(parent_chain({"name":input_text}))
}
with st.expander('Celebrity Name'):
        st.info(person_memory.buffer)
with st.expander('Celebrity Birth Place'):
        st.info(place_memory.buffer)
with st.expander('Celebrity State'):
        st.info(state_memory.buffer)