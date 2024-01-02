import os
import warnings
from typing import Dict
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from openfabric_pysdk.utility import SchemaUtil

from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import Ray, State
from openfabric_pysdk.loader import ConfigClass
import json



def config(configuration: Dict[str, ConfigClass], state: State):
    configuration["OPEN_API_KEY"] = ConfigClass('')


############################################################
# Callback function called on each execution pass
############################################################
    
def get_response(history,user_message,key,temperature=0):
   



    DEFAULT_TEMPLATE = """Imagine you are an AI expert in science, adept at answering complex science-related questions. In your response, please follow these steps:

    Break Down the Problem/Question: Start by explaining the question or problem in simpler terms. Make sure to clarify any complex scientific concepts or jargon involved.

    Brief Introduction: Provide a concise introduction to the topic. This should include essential background information that sets the stage for the detailed explanation.

    Mathematical Foundation (if applicable else skip): If the question involves mathematical concepts or requires calculations, lay out the mathematical foundation or equations that are relevant to solving the problem or understanding the concept.

    Scienctific reasoning: Use scientific reasoning to answer the question in depth by refering to mathematical figures and analysis

    Comprehensive Conclusion: Conclude with a thorough and well-rounded summary. Your conclusion should encapsulate the key points discussed and provide a clear answer or resolution to the original question.

    Remember, the aim is to make complex scientific concepts accessible and understandable, while providing accurate and detailed information.
    
    Relevant pieces of previous conversation:
    {context}


    (You do not need to use these pieces of information if not relevant)


    Current conversation:
    Human: {input}
    AI:"""

    
    
    PROMPT = PromptTemplate(
   input_variables=['context','input'], template=DEFAULT_TEMPLATE
    )



    chat_gpt = ChatOpenAI(temperature=temperature, model_name="gpt-3.5-turbo",openai_api_key=key)
    conversation_with_summary = LLMChain(
    llm=chat_gpt,
    prompt=PROMPT,
    verbose=False
    )
    response =conversation_with_summary.predict(context=history,input=user_message)
    return response

def get_class(user_message,key,temperature=0):
   



    DEFAULT_TEMPLATE = """

        Your role is to act as classifier. I want you to check if the message passed by user is regarding science or anything else. If the statement is about sciences return true in the a json format.  

        
        Examples:
        Exe1:
        user_input: 'Tell me about black holes'.
        OUTPUT: {{'result': True, 'score':0.99}}

        Exe2:

        user_input: 'Tell me about justin bieber.
        OUTPUT: {{'result': False, 'score':0.1}}


        Now, here is the user input:

        {input}

        """

    
    
    PROMPT = PromptTemplate(
   input_variables=['context','input'], template=DEFAULT_TEMPLATE
    )



    chat_gpt = ChatOpenAI(temperature=temperature, model_name="gpt-4",openai_api_key=key)
    conversation_with_summary = LLMChain(
    llm=chat_gpt,
    prompt=PROMPT,
    verbose=False
    )
    response =conversation_with_summary.predict(input=user_message)
    return response

def get_history(history_list):
    history=''
    for message in history_list:
        if message['role']=='user':
            history=history+'input '+message['content']+'\n'
        elif message['role']=='assistant':
            history=history+'output '+message['content']+'\n'
    
    return history
def execute(request: SimpleText, ray: Ray, state: State) -> SimpleText:
    output = []
    final_history=''
    key = ''
    for text in request.text:
        json_str=get_class(text,key)
        dict_like = eval(json_str)
        json_string = json.dumps(dict_like)


        status_dict=json.loads(json_string)

        if status_dict['result']:

            response=get_response(final_history,text,key)
            print('AI Response: ',response )

            print('-'*50)
            print()

            output.append({'role':'user','content':text})
            output.append({'role':'assistant','content':response})

            final_history=get_history(output)
        else:
            output.append({'role':'user','content':text})
            output.append({'role':'assistant','content':'I can only talk about science. Please ask me anything about it'})

        

    return SchemaUtil.create(SimpleText(), dict(text=output))
