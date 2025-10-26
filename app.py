import streamlit as st
from main_pipeline import get_answer  # use helper instead of chain directly

st.set_page_config(page_title="Graph RAG QA", layout="wide")
st.title("Graph RAG Question Answering")

question = st.text_input("Enter your question:")

if question:
    with st.spinner("Retrieving answer..."):
        response = get_answer(question)
        if response.startswith("⚠️ Error:"):
            st.error(response)
        else:
            st.success("Answer retrieved!")
            st.markdown("### Answer:")
            st.write(response)
