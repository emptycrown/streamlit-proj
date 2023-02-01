"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.llm_math.base import LLMMathChain
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool, load_tools
from langchain import SQLDatabase, SQLDatabaseChain
from gpt_index import GPTSimpleVectorIndex, WikipediaReader
from sqlalchemy import create_engine


# From here down is all the StreamLit UI.
st.set_page_config(
    page_title="Snowflake + Wikipedia + Langchain Demo", page_icon=":bird:"
)
st.header("Snowflake + Wikipedia + LLM Demo")
st.write(
    "👋 This is a demo of connecting large language models to external data sources to give it specialized knowledge (e.g. company transaction data) and reduce hallucinations."
)
st.write(
    "🤖 The chatbot is built with LangChain (agents) and GPT Index (connect to data sources)."
)

st.write("Examples you can try:")
st.write("- What was the average size of transactions in January?")
st.write("- How much did Bill Gates spend on transactions and where did he grow up?")
st.write("- What was the largest transaction? Who made that transaction?")
st.write(
    "- Who were the celebrities that purchased, and how much did they spend in total?"
)
st.write("- Did Bill Gates or Elon Musk spend more relative to their net worth?")

st.sidebar.title("Data Sources")

llm = OpenAI(temperature=0)

# Connect to Snowflake and build the chain
@st.experimental_singleton
def build_snowflake_chain():
    engine = create_engine(
        "snowflake://{user}:{password}@{account}/{database}/{schema}?warehouse={warehouse}".format(
            **st.secrets["snowflake"]
        )
    )

    sql_database = SQLDatabase(engine)

    st.sidebar.header("❄️ Snowflake database has been connected")
    st.sidebar.write(f"{sql_database.table_info}")

    db_chain = SQLDatabaseChain(llm=llm, database=sql_database)
    return db_chain


# Parse and Index the Wiki Pages
@st.experimental_singleton
def build_index(input):
    pages = [p.strip() for p in input.split(",") if p != ""]
    wiki_docs = WikipediaReader().load_data(pages=pages) if input else []
    return GPTSimpleVectorIndex(wiki_docs), pages


# Snowflake tool
db_chain = build_snowflake_chain()

st.sidebar.write("")

# Wiki tool
st.sidebar.header("📚 You can also add Wikipedia pages")
wiki_input = st.sidebar.text_input(
    "Comma-separated Wiki pages: ", placeholder="e.g. Tokyo, Berlin, Rome", key="wiki"
)

index, wiki_pages = build_index(wiki_input)

if len(wiki_pages) > 0:
    st.sidebar.write(f"{len(wiki_pages)} articles have been parsed and indexed")

tools = [
    Tool(
        name="Snowflake Transactions",
        func=lambda q: db_chain.run(q),
        description=f"Useful when you want to answer questions about people's spending, purchases, and transactions. The input to this tool should be a complete english sentence. The celebrities are: Ruth Porat, Bill Gates, Warren Buffet, Elon Musk, Susan Wojcicki.",
    ),
    Tool(
        name="Wiki GPT Index",
        func=lambda q: str(index.query(q, similarity_top_k=1)),
        description=f"Useful when you want to answer general knowledge and trivia questions about notable figures and net worth. If this tool is used, only explicitly pass in what original query is. The input to this tool should be a complete english sentence.",
    ),
    Tool(
        "Calculator",
        LLMMathChain(llm=llm).run,
        "Useful for when you need to make any math calculations. Use this tool for any and all numerical calculations. The input to this tool should be a mathematical expression.",
    ),
]


# Initialize LangChain agent and chain

memory = ConversationBufferMemory(memory_key="chat_history")

agent_chain = initialize_agent(
    tools, llm, agent="zero-shot-react-description", verbose=True, memory=memory
)


def refresh_chain():
    """Refresh the chain variables.."""
    print("refreshing the chain")
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["buffer"] = []
    print("chain refreshed")


if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


user_input = st.text_input("You: ", placeholder="Hello!", key="input")

if user_input:
    output = agent_chain.run(input=user_input)
    # output = index.query(user_input).response
    if not st.session_state["past"]:
        st.session_state["past"] = []
    if not st.session_state["generated"]:
        st.session_state["generated"] = []
    st.session_state["past"].append(user_input)
    st.session_state["generated"].append(output)


if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        message(st.session_state["generated"][i], key=str(i))


st.button("Clear chat", on_click=refresh_chain)

# if chain.memory.store:
#     for entity, summary in chain.memory.store.items():
#         # st.sidebar.write(f"{entity}: {summary}")
#         st.sidebar.write(f"Entity: {entity}")
#         st.sidebar.write(f"{summary}")
