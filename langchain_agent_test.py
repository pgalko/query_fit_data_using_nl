

#----------------ENVIRONMENT VARIABLES----------------

import os

# Openai_API_key
api_key = os.environ.get('OPENAI_API_KEY')
# Serper API key
serper_api_key = os.environ.get('SERPER_API_KEY')

#----------------PARAMETERS (Path to PDF document, sample prompt----------------

# The prompt that will be used to generate a response from the agent
prompt = " Get the formula for the HR-Running-Speed-Index, then plug in the mean hr and mean speed data from the df, and the standing hr of 56, maximal hr of 192, and the vo2max running speed of 4.5 m/s. Calculate the results and return the HR-Running-Speed-Index together with the formula used."

# The path to the local PDF document that will be used to provide a context for the agent
document_path =  "Heart_Rate_Running_Speed_Index_May_Be_an_Efficient.4.pdf"

#----------------COMPONENTS (Models,Embeddings,Wrappers) ----------------

from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.utilities import GoogleSerperAPIWrapper

# Create an LLM object
llm = OpenAI(temperature=0,openai_api_key=api_key,verbose=True)
# Create an embeddings object
embeddings = OpenAIEmbeddings()
# Create a wrapper for the Google Serper API
search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key, verbose=True)

#----------------DATAFRAME----------------

# Create a dataframe to use as an input to the agent.
import pandas as pd

# Read the data from csv files
df = pd.read_csv('test_activity_data.csv')
# Convert datetime column to datetime format
df['datetime'] = pd.to_datetime(df['datetime'])
# Drop NaN values
df = df.dropna()

#----------------IN-MEMORY VECTOR DB----------------

from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredPDFLoader

# Load the PDF document and split it into chunks
loader = UnstructuredPDFLoader(document_path)
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

# Create a vector store object and ingest and vectorize the documents
vector_store = Chroma.from_documents(texts, embeddings, collection_name="pdf-doc-search")

# Create a retriever object
retriever = vector_store.as_retriever()

# Print the result of the search query
print(retriever.get_relevant_documents(prompt))

# Create a chain object. This is used to create the tool that will be used by the agent.
docs_db = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, verbose=True)

#----------------AGENT----------------

from typing import Any, List, Optional
from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.pandas.prompt import PREFIX, SUFFIX
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.agents import Tool
import pandas as pd

# Create Agent, and specify tools that the agent wil have access to
def pandas_dataframe_agent(
    llm: BaseLLM,
    df: Any,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = PREFIX,
    suffix: str = SUFFIX,
    input_variables: Optional[List[str]] = None,
    verbose: bool = True,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a pandas agent from an LLM and dataframe."""
    
    # Check if the input is a dataframe
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas object, got {type(df)}")
    if input_variables is None:
        input_variables = ["df", "input", "agent_scratchpad"]
        
    # Specify the tools that the agent will have access to   
    tools = [
        PythonAstREPLTool(
                         name="pandas_df",
                         locals={"df": df},
                         description=" use this tool to query the Pandas Dataframe and manipulate the output.",
                         ),
        Tool(
            name = "pdf-doc-search",
            func=docs_db.run,
            description=" use this tool to query the documents stored in the local Chroma vector DB."
            ),
        Tool(
            name = "google-search",
            func=search.run,
            description=" use this tool to search Google using Serper API.",
            )
    ]
    
    # Create the prompt that will be used to generate a response from the agent
    prompt = ZeroShotAgent.create_prompt(
        tools, prefix=prefix, suffix=suffix, input_variables=input_variables
    )
    
    partial_prompt = prompt.partial(df=str(df.head()))
    llm_chain = LLMChain(
        llm=llm,
        prompt=partial_prompt,
        callback_manager=callback_manager,
    )
    
    # Specify the allowed tools that the agent will have access to
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)
    
    # Return the agent
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=verbose)

# Here is where the magic happens :-). The dataframe and the document are passed to the agent, which is then used to generate a response.
agent = pandas_dataframe_agent(llm, df, verbose=True)

# Generate a response from the agent
response = agent.run(prompt)

