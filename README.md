# Snowflake + Wikipedia + LLM Demo

This is a small Streamlit app for the [Replit Bounty](https://twitter.com/altcap/status/1620446003777404930?s=20) posted by Altimeter Capital. It connects Wikipedia articles and a Snowflake database to GPT so that the model has additional knowledge when answering questions. The data is connected using GPT Index and a LangChain agent is used to determine which data source is used for a specific query.

Run it locally with:
```
streamlit run main.py
```
