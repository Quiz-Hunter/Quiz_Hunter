from SimilaritySearcher import SimilaritySearcher
import glob

def input_question():
    print("\n🔹 題目/題組輸入 🔹\n")

    context = input("請輸入題組內容 (若無則直接 Enter)：\n").strip()
    stem = input("請輸入題目內容 (必填)：\n").strip()

   
    return context, stem

def main():
    print("\n相似題目比對系統\n")

    # 自動載入當前目錄下所有npz向量檔案
    npz_files = glob.glob("./*.npz")
    if not npz_files:
        print(" 無法找到任何 .npz 向量檔案，請確認後再試一次。")
        return

    # 載入向量檔
    searcher = SimilaritySearcher(npz_files)

    while True:
        context, stem = input_question()
        print("\n正在進行相似度分析...\n")

        # 執行搜尋
        searcher.search(context, stem, top_k=5)

        cont = input("是否繼續下一個搜尋？(y/n)：").strip().lower()
        if cont != 'y':
            print("已結束搜尋，感謝使用！")
            break

if __name__ == "__main__":
    main()
