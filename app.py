import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"  # 關閉熱重載，避免 torch._classes bug

import streamlit as st
import core.RetrieverUtils as retriever
import torch
import core.Score as rank_score

@st.cache_resource
def load_retriever_system():
    return retriever.bm25_hnsw_retriever()

def interface():
    st.set_page_config(page_title="QuizHunter", layout="wide")
    st.title("QuizHunter Chatbot")

    query = st.text_input("請輸入想要查找的類似問題 👇")

    # 初始化 Session State
    if "last_query" not in st.session_state:
        st.session_state.last_query = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_score" not in st.session_state:
        st.session_state.last_score = None

    # 如果輸入了新查詢
    if query and query != st.session_state.last_query:
        with st.spinner("正在分析與推薦中，請稍候..."):
            llm_retriever = load_retriever_system()

            results = llm_retriever.search(query, top_k=3, alpha=0.5)
            first_result = results[0]

            st.session_state.last_query = query
            st.session_state.last_result = first_result

            try:
                stars, gold, answers, correctness, auto, gem = rank_score.DifficultyScorer(first_result).score()
                st.session_state.last_score = (stars, gold, auto, gem)
            except Exception as e:
                st.error("⚠️ 無法取得 Gemini 回應，可能已超出配額或速率限制，請稍候再試。")
                st.exception(e)  # 若你要顯示原始錯誤訊息（可選）
                return

    # 顯示查詢結果與評分
    if st.session_state.last_result:
        r = st.session_state.last_result
        stars, gold, auto, gem = st.session_state.last_score

        st.markdown("### 🔍 查詢結果")
        st.write(r['content'])

        stars_mark = '⭐️' * stars
        st.markdown(f"**預估難度：** {stars_mark}")
        st.success("✅ 回答完成")

        if st.button("🔍 顯示正解"):
            st.markdown(f"**🔑 正確解答：** {gold}")

    # 年度圖像分析區
    st.markdown("---")
    st.subheader("📊 年度題目分析結果")
    selected_year = st.selectbox("請選擇年份", [str(y) for y in range(106, 114)], index=7)

    wordcloud_path = os.path.join("results", "wordclouds", f"{selected_year}.png")
    keywords_path  = os.path.join("results", "keywords",  f"{selected_year}.png")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("☁︎ 題目詞雲分析")
        if os.path.exists(wordcloud_path):
            st.image(wordcloud_path, caption=f"{selected_year} 年詞雲圖", use_container_width=True)
        else:
            st.warning(f"找不到詞雲圖片：{wordcloud_path}")

    with col2:
        st.markdown("🔑 關鍵字頻率分析")
        if os.path.exists(keywords_path):
            st.image(keywords_path, caption=f"{selected_year} 年關鍵字分析圖", use_container_width=True)
        else:
            st.warning(f"找不到關鍵字圖片：{keywords_path}")

def main():
    interface()

if __name__ == '__main__':
    main()
