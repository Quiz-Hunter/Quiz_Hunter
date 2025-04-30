import crawler.hackmd_crawler as HackmdCrawler
import streamlit as st
import model.rag_model as Rag
import model.summarize_model as Summarize_RAG
import torch

@st.cache_resource
def load_rag_system():
    llm_rag = Rag.RagSystem('./doc')
    llm_rag.build_index()
    return llm_rag

def interface():
    st.set_page_config(page_title="ACLAB 財務 RAG Chatbot", layout="wide")
    st.title("🤖 ACLAB 財務 RAG Chatbot ")

    query = st.text_input("請輸入你的問題 👇")

    if query:
        with st.spinner("正在分析與回答中，請稍候..."):
            if "帳號" in query or "帳密" in query:
                st.success("✅ 回答完成")
                st.write("帳密為機密訊息，請您去此頁面查找 https://hackmd.io/Ai08e_pVTCeKJ78QbiDJ6Q?view")
            else:
                if "國外差旅" in query or "機票" in query:
                    st.write("示範影片連結 https://drive.google.com/file/d/1nftAzgKWsrZ9Wr9CNXaM0kMynMMgETcN/view?usp=drive_link")
                    st.write("日支費注意事項 https://drive.google.com/file/d/1bfPjs_FrI1MJPeKEqb2U_BsrlPq7DD09/view?usp=drive_link")
                
                # llm_summarize = Summarize_RAG.SummarizationSystem('./doc')
                llm_rag = load_rag_system()
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