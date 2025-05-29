import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"  # 關閉熱重載，避免 torch._classes bug

import streamlit as st
import core.RetrieverUtils as retriever
import torch

@st.cache_resource
def load_retriever_system():
    bm25_retriever = retriever.bm25_hnsw_retriever("test.csv")
    return bm25_retriever

def interface():
    st.set_page_config(page_title="QuizHunter", layout="wide")
    st.title("QuizHunter Chatbot")

    query = st.text_input("請輸入想要查找的類似問題 👇")

    if query:
        with st.spinner("正在分析與推薦中，請稍候..."):
            llm_retriever = load_retriever_system()
            llm_retriever.load_and_prepare()
            results = retriever.search(query, top_k=3, alpha=0.5)
            for r in results:
                print(f"{r['id']} | {r['date']} | {r['score']:.2f} | {r['content']}")
            st.success("✅ 回答完成")
            st.write(results[0]['content'])

    # 🔽 統一選擇年份
    st.markdown("---")
    st.subheader("📊 年度題目分析結果")
    selected_year = st.selectbox("請選擇年份", [str(y) for y in range(106, 114)], index=7)

    # 🔄 圖片路徑
    wordcloud_path = os.path.join("results", "wordclouds", f"{selected_year}.png")
    keywords_path  = os.path.join("results", "keywords",  f"{selected_year}.png")

    # 🔄 顯示圖片，兩圖同列排版
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("☁︎ 題目詞雲分析")
        if os.path.exists(wordcloud_path):
            st.image(wordcloud_path, caption=f"{selected_year} 年詞雲圖", use_container_width=True)
        else:
            st.warning(f"找不到詞雲圖片：results/wordclouds/{selected_year}.png")

    with col2:
        st.markdown("🔑 關鍵字頻率分析")
        if os.path.exists(keywords_path):
            st.image(keywords_path, caption=f"{selected_year} 年關鍵字分析圖", use_container_width=True)
        else:
            st.warning(f"找不到關鍵字圖片：results/keywords/{selected_year}.png")

def main():
    interface()

if __name__ == '__main__':
    main()
