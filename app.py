import crawler.hackmd_crawler as HackmdCrawler
import streamlit as st
import model.rag_model as Rag
import model.summarize_model as Summarize_RAG
import torch

@st.cache_resource
def load_retriever_system():
    llm_rag = Rag.RagSystem('./doc')
    llm_rag.build_index()
    return llm_rag

def interface():
    st.set_page_config(page_title="QuizHunter", layout="wide")
    st.title("QuizHunter Chatbot ")

    query = st.text_input("請輸入想要查找的類似問題 👇")

    if query:
        with st.spinner("正在分析與推薦中，請稍候..."):
            llm_rag = load_retriever_system()
            # llm_summarize.build_index()
            response , content =llm_rag.answer(query)
            # response =llm_summarize.summarize(query)
            st.success("✅ 回答完成")
            # st.write(content)
            st.write(response)

def main():
    # hack_md_crawler = HackmdCrawler.HackmdCrawler('./config.env')
    # hack_md_crawler.crawl()
    # print(torch.cuda.is_available())
    interface()


if __name__ == '__main__':
    main()