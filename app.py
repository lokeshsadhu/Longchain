# Conversational Q&A chatbot
import streamlit as st
from langchain.schema import HumanMessage,SystemMessage,AIMessage
from langchain_openai import ChatOpenAI

## Streamlit UI
st.set_page_config(page_title="Personalized order Chatbot")
st.header("Hey, Customer")

from dotenv import load_dotenv
load_dotenv()
import os

chat=ChatOpenAI(temperature=0.7)

if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages']=[
        SystemMessage(content="""
            You are OrderBot, an automated service to collect orders for a pizza restaurant. \
            You first greet the customer, then collects the order, \
            and then asks if it's a pickup or delivery. \
            You wait to collect the entire order, then summarize it and check for a final \
            time if the customer wants to add anything else. \
            If it's a delivery, you ask for an address. \
            Finally you collect the payment.\
            Make sure to clarify all options, extras and sizes to uniquely \
            identify the item from the menu.\
            You respond in a short, very conversational friendly style. \
            The menu includes \
            pepperoni pizza  12.95, 10.00, 7.00 \
            cheese pizza   10.95, 9.25, 6.50 \
            eggplant pizza   11.95, 9.75, 6.75 \
            fries 4.50, 3.50 \
            greek salad 7.25 \
            Toppings: \
            extra cheese 2.00, \
            mushrooms 1.50 \
            sausage 3.00 \
            canadian bacon 3.50 \
            AI sauce 1.50 \
            peppers 1.00 \
            Drinks: \
            coke 3.00, 2.00, 1.00 \
            sprite 3.00, 2.00, 1.00 \
            bottled water 5.00 \
            after the conversation at last create a json summary of food order. Itemize the price for each item\
 The fields should be 1) pizza, include size 2) list of toppings 3) list of drinks, include size   4) list of sides include size  5)total price '},    
)
 #The fields should be 1) pizza, price 2) list of toppings 3) list of drinks, include size include price  4) list of sides include size include price, 5)total price '},
 diplay this summary after the order in json format only  
            """)
    ]

## Function to load OpenAI model and get respones

def get_chatmodel_response(question):

    st.session_state['flowmessages'].append(HumanMessage(content=question))
    answer=chat(st.session_state['flowmessages'])
    st.session_state['flowmessages'].append(AIMessage(content=answer.content))
    return answer.content

input=st.text_input("Input: ",key="input")
response=get_chatmodel_response(input)
# submit=st.button("Ask the question")

## If ask button is clicked

if input:
    st.subheader("The Response is")
    st.write(response)

    
