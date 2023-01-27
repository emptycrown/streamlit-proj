"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool

from gpt_index import GPTSimpleVectorIndex, WikipediaReader
from gpt_index.langchain_helpers.memory_wrapper import GPTIndexMemory

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Wikipedia + Langchain Demo", page_icon=":bird:")
st.header("Wikipedia + Langchain Demo")

st.sidebar.title("Data Sources")
wiki_input = st.sidebar.text_input(
    "Comma-separated Wiki pages: ", placeholder="e.g. Tokyo, Berlin, Rome", key="wiki"
)


@st.cache(allow_output_mutation=True)
def build_index(input):
    pages = [p.strip() for p in input.split(",") if p != ""]
    wiki_docs = WikipediaReader().load_data(pages=pages) if input else []
    return GPTSimpleVectorIndex(wiki_docs), pages


index, wiki_pages = build_index(wiki_input)

if len(wiki_pages) > 0:
    st.sidebar.write(f"{len(wiki_pages)} articles have been parsed and indexed")

tools = [
    Tool(
        name="GPT Index",
        func=lambda q: str(index.query(q).response),
        description=f"useful when you want to answer questions from the topics {wiki_input}. The input to this tool should be a complete english sentence.",
        return_direct=True,
    ),
]

# st.sidebar.text_input(
#     "Snowflake", placeholder="This is not working yet", key="snowflake"
# )

# Initialize LangChain agent and chain

memory = ConversationBufferMemory(memory_key="chat_history")
llm = OpenAI(temperature=0)
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
