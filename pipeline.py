import os, json
from pathlib import Path
from core.QuestionExtractor import QuestionExtractor
from core.EmbeddingGenerator import EmbeddingGenerator
from core.SimilaritySearcher import SimilaritySearcher

def run_pipeline(pdf_folder="pdf_data", output_folder="output_data", year_start=106, year_end=113):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    json_paths, npz_paths = [], []

    # Step 1: Extract questions from PDFs
    for year in range(year_start, year_end + 1):
        pdf_path = os.path.join(pdf_folder, f"{year}_q.pdf")
        json_path = os.path.join(output_folder, f"{year}.json")
        npz_path = os.path.join(output_folder, f"{year}.npz")

        print(f"\n📄 處理中：{pdf_path}")
        extractor = QuestionExtractor(pdf_path)
        extractor.process_pdf(json_path)
        json_paths.append(json_path)

        # Step 2: Generate embeddings
        embedder = EmbeddingGenerator(json_path)
        embedder.generate_embeddings(npz_path)
        npz_paths.append(npz_path)

    print("\n✅ 所有 PDF 處理與 Embedding 完成")

    # Step 3 (optional): Search similar questions
    print("\n🔍 初始化 Similarity Searcher...")
    searcher = SimilaritySearcher(npz_paths)

    while True:
        try:
            year = input("\n🔎 輸入要查找相似題目的年分 (例如：108)，或輸入 q 離開：")
            if year.lower() == 'q':
                break
            json_path = os.path.join(output_folder, f"{year}.json")
            if not Path(json_path).exists():
                print("⚠️ JSON 檔案不存在。")
                continue
            with open(json_path, encoding="utf-8") as f:
                questions = json.load(f)
            qid = int(input(f"👉 輸入想查詢的題號 (1 ~ {len(questions)}): "))
            if not (1 <= qid <= len(questions)):
                print("⚠️ 題號無效。")
                continue
            q = questions[qid - 1]
            searcher.search(q.get("group_context", ""), q["stem"], top_k=5)
        except Exception as e:
            print("❌ 發生錯誤：", e)

if __name__ == "__main__":
    run_pipeline()
