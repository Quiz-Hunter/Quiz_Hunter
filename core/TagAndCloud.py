import json
import jieba
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer


class QuizAnalyzer:
    def __init__(self,
                 stopwords_path: str = None,
                 keybert_model: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.stopwords = set()
        if stopwords_path:
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                self.stopwords = {w.strip() for w in f if w.strip()}
        self.kb = KeyBERT(keybert_model)

    def tokenize(self, text: str) -> list[str]:
        return [w for w in jieba.lcut(text)
                if w.strip() and w not in self.stopwords and len(w.strip()) > 1]

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

    def extract_tags_keybert(self,
                             texts: list[str],
                             top_k: int = 5) -> list[list[str]]:
        all_tags = []
        for txt in texts:
            tokens = self.tokenize(txt)
            joined = ' '.join(tokens)
            kws = self.kb.extract_keywords(
                joined,
                keyphrase_ngram_range=(1, 2),
                stop_words=None,
                top_n=top_k
            )
            all_tags.append([kw for kw, _ in kws])
        return all_tags

    def tag_json_and_save(self,
                          input_json: str,
                          output_json: str,
                          top_k: int = 5) -> None:
        questions, raw = self.load_questions_from_json(input_json)
        tags = self.extract_tags_keybert(questions, top_k=top_k)
        all_tags_flat = []
        for item, tg in zip(raw, tags):
            item['tags'] = tg
            all_tags_flat.extend(tg)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 已將標籤寫入 {output_json}")

        # 顯示 top 10 tag 長條圖
        self.plot_top_tags(all_tags_flat)

    def plot_top_tags(self, tags: list[str], top_n: int = 10):
        counter = Counter(tags)
        common = counter.most_common(top_n)
        tags, counts = zip(*common)
        plt.figure(figsize=(10, 6))
        plt.bar(tags, counts, color='skyblue')
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Top {top_n} Tags')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

    def generate_wordcloud_tfidf(self,
                                 texts: list[str],
                                 font_path: str = 'NotoSansTC-Regular.otf',
                                 top_k: int = None,
                                 output_path: str = None) -> None:
        corpus = [' '.join(self.tokenize(t)) for t in texts]
        vect = TfidfVectorizer()
        mat = vect.fit_transform(corpus)
        names = vect.get_feature_names_out()
        scores = mat.sum(axis=0).A1
        tfidf_dict = dict(zip(names, scores))
        if top_k:
            tfidf_dict = dict(sorted(tfidf_dict.items(),
                                     key=lambda x: x[1],
                                     reverse=True)[:top_k])
        wc = WordCloud(font_path=font_path,
                       background_color='white',
                       width=800,
                       height=600)
        wc.generate_from_frequencies(tfidf_dict)

        if output_path:
            wc.to_file(output_path)
            print(f"[INFO] 詞雲已儲存至 {output_path}")
        else:
            plt.figure(figsize=(10, 8))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout()
            plt.show()

    def process_and_visualize(self,
                              json_path: str,
                              font_path: str = 'NotoSansTC-Regular.otf',
                              top_k: int = None,
                              output_path: str = None) -> None:
        texts, _ = self.load_questions_from_json(json_path)
        self.generate_wordcloud_tfidf(
            texts,
            font_path=font_path,
            top_k=top_k,
            output_path=output_path
        )


if __name__ == '__main__':
    qa = QuizAnalyzer(stopwords_path='stopwords.txt')

    qa.tag_json_and_save(
        input_json='questions.json',
        output_json='questions_with_tags.json',
        top_k=3
    )

    qa.process_and_visualize(
        json_path='questions.json',
        font_path='NotoSansTC-Regular.otf',
        top_k=100
    )
