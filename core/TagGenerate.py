import json
import jieba
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from keybert import KeyBERT
from matplotlib import font_manager
from sklearn.feature_extraction.text import TfidfVectorizer
import os


class QuizAnalyzer:
    def __init__(self, stopwords_path=None, keybert_model='paraphrase-multilingual-MiniLM-L12-v2'):
        self.stopwords = set()
        if stopwords_path:
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                self.stopwords = {w.strip() for w in f if w.strip()}

        # ✅ 加入內建雜訊詞（過濾掉「何者」、「正確」、「應選」等）
        self.builtin_junk = {
            '何者', '正確', '錯誤', '以下', '哪些', '有關', '關於', '描述',
            '應選', '符合', '選項', '依據', '資料', '圖', '表', '請問',
            '是何者', '為何', '敘述', '應為', '選出', '判斷', '說明'
        }

        self.kb = KeyBERT(keybert_model)

    def tokenize(self, text: str) -> list[str]:
        return [
            w for w in jieba.lcut(text)
            if w.strip() and w not in self.stopwords and w not in self.builtin_junk and len(w.strip()) > 1
        ]

    def load_questions_from_json(self, json_path: str) -> tuple[list[str], list[dict]]:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        questions = []
        for item in data:
            if 'stem' in item:
                combined = ''
                if item.get('group_context'):
                    combined += item['group_context'].strip() + ' '
                combined += item['stem'].strip()
                questions.append(combined)
        return questions, data

    def extract_tags_keybert(self, texts: list[str], top_k: int = 15) -> list[list[str]]:
        all_tags = []
        for txt in texts:
            tokens = self.tokenize(txt)
            if not tokens:
                all_tags.append([])
                continue
            joined = ' '.join(tokens)
            kws = self.kb.extract_keywords(
                joined,
                keyphrase_ngram_range=(1, 2),
                stop_words=None,
                top_n=top_k
            )
            filtered = [kw for kw, _ in kws if kw and len(kw) > 1 and kw not in self.stopwords]
            all_tags.append(filtered)
        return all_tags


    def tag_json_and_save(self, input_json: str, output_json: str, top_k: int = 15) -> list[str]:
        questions, raw = self.load_questions_from_json(input_json)
        tags = self.extract_tags_keybert(questions, top_k=top_k)

        all_tags_flat = []
        for item, tag_list in zip(raw, tags):
            item['tags'] = tag_list
            all_tags_flat.extend(tag_list)  # ✅ 加入所有 tag，不管是否重複

        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)

        return all_tags_flat  # ✅ 返回累積過的 tag 列表供統計用


    def plot_top_tags(self, tags: list[str], output_path: str, top_n: int = 10, font_path: str = None):
        counter = Counter(tags)
        common = counter.most_common(top_n)
        if not common:
            print(f"[WARN] 無熱門標籤可繪製: {output_path}")
            return

        tag_names, counts = zip(*common)

        # ✅ 設定中文字型
        font_prop = None
        if font_path:
            font_prop = font_manager.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
        else:
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Noto Sans TC', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False  # 修正負號顯示

        # ✅ 畫圖
        plt.figure(figsize=(10, 6))
        plt.bar(tag_names, counts, color='skyblue')
        plt.xticks(rotation=45, ha='right', fontproperties=font_prop)
        plt.title(f'Top {top_n} Tags', fontproperties=font_prop)
        plt.ylabel('Count', fontproperties=font_prop)
        plt.tight_layout()
        plt.savefig(output_path, format='jpg')
        plt.close()
        print(f"[INFO] 熱門標籤圖儲存於 {output_path}")

    def generate_wordcloud_tfidf(self,
                                 texts: list[str],
                                 font_path: str,
                                 top_k: int,
                                 output_path: str) -> None:
        corpus = [' '.join(self.tokenize(t)) for t in texts]
        vect = TfidfVectorizer()
        mat = vect.fit_transform(corpus)
        names = vect.get_feature_names_out()
        scores = mat.sum(axis=0).A1
        tfidf_dict = dict(zip(names, scores))
        if top_k:
            tfidf_dict = dict(sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:top_k])
        wc = WordCloud(font_path=font_path,
                       background_color='white',
                       width=800,
                       height=600)
        wc.generate_from_frequencies(tfidf_dict)
        wc.to_file(output_path)
        print(f"[INFO] 詞雲圖儲存於 {output_path}")

    def process_year_file(self, year: int,
                          font_path: str = 'NotoSansTC-Regular.otf',
                          top_k_wordcloud: int = 100,
                          top_k_tags: int = 15):
        input_json = f"./Quiz_json/{year}.json"
        output_json = f"./results/{year}_with_tags.json"
        wordcloud_path = f"./results/wordclouds/{year}.png"
        tag_bar_path = f"./results/keywords/{year}.png"

        print(f"[INFO] 處理中: {input_json}")
        questions, _ = self.load_questions_from_json(input_json)

        # 詞雲圖
        self.generate_wordcloud_tfidf(questions, font_path, top_k_wordcloud, wordcloud_path)

        # 標籤統計
        all_tags = self.tag_json_and_save(input_json, output_json, top_k=top_k_tags)
        self.plot_top_tags(all_tags, output_path=tag_bar_path)


if __name__ == '__main__':
    qa = QuizAnalyzer(stopwords_path='./core/stopwords.txt')

    for year in range(106, 114):
        if year == 112: continue
        qa.process_year_file(
            year=year,
            font_path='C:/Windows/Fonts/msjh.ttc',
            top_k_wordcloud=100,
            top_k_tags=15
        )

