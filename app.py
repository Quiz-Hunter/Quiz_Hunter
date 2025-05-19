import streamlit as st
import core.retriever_utils as retriever
import torch

@st.cache_resource
def load_retriever_system():
    bm25_retriever = retriever.bm25_hnsw_retriever("test.csv")
    return bm25_retriever

def interface():
    st.set_page_config(page_title="QuizHunter", layout="wide")
    st.title("QuizHunter Chatbot ")

    query = st.text_input("請輸入想要查找的類似問題 👇")

    if query:
        with st.spinner("正在分析與推薦中，請稍候..."):
            llm_retriever = load_retriever_system()
            llm_retriever.load_and_prepare()
            # llm_summarize.build_index()
            results = retriever.search(query, top_k=3, alpha=0.5)
            for r in results:
                print(f"{r['id']} | {r['date']} | {r['score']:.2f} | {r['content']}")
            # response =llm_summarize.summarize(query)
            st.success("✅ 回答完成")
            # st.write(content)
            st.write(results[0]['content'])

def main():
    # hack_md_crawler = HackmdCrawler.HackmdCrawler('./config.env')
    # hack_md_crawler.crawl()
    # print(torch.cuda.is_available())
    interface()


if __name__ == '__main__':
    main()