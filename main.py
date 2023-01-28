"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain import SQLDatabase, SQLDatabaseChain

from gpt_index import GPTSimpleVectorIndex, WikipediaReader
from gpt_index.langchain_helpers.memory_wrapper import GPTIndexMemory

# import snowflake.connector
from sqlalchemy import create_engine


# From here down is all the StreamLit UI.
st.set_page_config(
    page_title="Snowflake + Wikipedia + Langchain Demo", page_icon=":bird:"
)
st.header("Snowflake + Wikipedia + Langchain Demo")

st.sidebar.title("Data Sources")

llm = OpenAI(temperature=0)

# Initialize connection.
# Uses st.experimental_singleton to only run once.
# @st.experimental_singleton
# def init_connection():
#     return snowflake.connector.connect(
#         **st.secrets["snowflake"], client_session_keep_alive=True
#     )


# conn = init_connection()

# @st.experimental_memo(ttl=600)
# def run_query(query):
#     with conn.cursor() as cur:
#         cur.execute(query)
#         return cur.fetchall()


# rows = run_query("SELECT * from mytable;")
# # Print results.
# for row in rows:
#     st.sidebar.write(f"{row[0]} has a :{row[1]}:")


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

    db_chain = SQLDatabaseChain(llm=llm, database=sql_database, verbose=True)
    return db_chain


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
        name="Wiki GPT Index",
        func=lambda q: str(index.query(q)),
        description=f"useful when you want to answer questions from the topics {wiki_input}. The input to this tool should be a complete english sentence.",
        return_direct=True,
    ),
    Tool(
        name="Snowflake SQL Chain",
        func=lambda q: db_chain.run(q),
        description=f"useful when you want to answer questions by looking up data in the Snowflake transaction database using SQL. The input to this tool should be a complete english sentence.",
        return_direct=True,
    ),
]


# Initialize LangChain agent and chain

memory = ConversationBufferMemory(memory_key="chat_history")

agent_chain = initialize_agent(
    tools, llm, agent="conversational-react-description", verbose=True, memory=memory
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


st.button("Refresh chat", on_click=refresh_chain)

# if chain.memory.store:
#     for entity, summary in chain.memory.store.items():
#         # st.sidebar.write(f"{entity}: {summary}")
#         st.sidebar.write(f"Entity: {entity}")
#         st.sidebar.write(f"{summary}")
