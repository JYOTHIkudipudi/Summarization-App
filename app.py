import openai
import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI

openai.api_key = os.environ['OPENAI_API_KEY']

@st.cache_data
def setup_documents(pdf_file_path, chunk_size, chunk_overlap):
    loader = PyPDFLoader(pdf_file_path)
    docs_raw = loader.load()
    docs_raw_text = [doc.page_content for doc in docs_raw]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.create_documents(docs_raw_text)
    return docs

def custom_summary(docs,llm, custom_prompt, chain_type, num_summaries):
    custom_prompt = custom_prompt + """:\n\n {text}"""
    COMBINE_PROMPT = PromptTemplate(template=custom_prompt, input_variables=["text"])
    MAP_PROMPT = PromptTemplate(template="Summarize:\n\n{text}", input_variables=["text"])

    if chain_type == "map_reduce":
        chain = load_summarize_chain(llm, chain_type=chain_type,
                                     map_prompt=MAP_PROMPT,
                                     combine_prompt=COMBINE_PROMPT)
    else:
        chain = load_summarize_chain(llm, chain_type=chain_type)

    summaries = []

    for i in range(num_summaries):
        summary_output = chain(
            {"input_documents": docs},
            return_only_outputs=True
        )["output_text"]

        summaries.append(summary_output)

    return summaries


def main():
    st.title("AI PDF Summarizer")

    chain_type = st.sidebar.selectbox(
        "Chain Type",
        ["map_reduce", "stuff", "refine"]
    )

    chunk_size = st.sidebar.slider(
        "Chunk Size",
        100,
        10000,
        1900
    )

    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap",
        100,
        10000,
        200
    )

    user_prompt = st.text_input("Enter Custom Prompt")
    pdf_file_path = st.text_input("Enter PDF File Path")

    temperature = st.sidebar.slider(
        "Temperature",
        0.0,
        1.0,
        0.0
    )

    num_summaries = st.sidebar.slider(
        "Number of Summaries",
        1,
        5,
        1
    )

    llm_choice = st.sidebar.selectbox(
        "Model",
        ["ChatGPT", "GPT4"]
    )

    if llm_choice == "ChatGPT":
        llm = ChatOpenAI(temperature=temperature)

    else:
        llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=temperature
        )

    if pdf_file_path != "":
        docs = setup_documents(
            pdf_file_path,
            chunk_size,
            chunk_overlap
        )

        st.success("PDF Loaded Successfully")

        if st.button("Summarize"):

            results = custom_summary(
                docs,
                llm,
                user_prompt,
                chain_type,
                num_summaries
            )

            st.subheader("Generated Summaries")

            for summary in results:
                st.write(summary)


if __name__ == "__main__":
    main()
