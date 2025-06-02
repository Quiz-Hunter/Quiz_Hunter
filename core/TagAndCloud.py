import json
import jieba
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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
                keyphrase_ngram_range=(1, 1),
                stop_words=None,
                top_n=top_k
            )
            all_tags.append([kw for kw, _ in kws])
        return all_tags

    def tag_json_and_save(self,
                          input_json: str,
                          output_path: str,
                          top_k: int = 5) -> None:
        questions, raw = self.load_questions_from_json(input_json)
        tags = self.extract_tags_keybert(questions, top_k=top_k)
        all_tags_flat = []
        for item, tg in zip(raw, tags):
            item['tags'] = tg
            all_tags_flat.extend(tg)
        if output_path:
            output_json = output_path + '.json'
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 已將標籤寫入 {output_json}")

        # 顯示 top 10 tag 長條圖
        self.plot_top_tags(tags=all_tags_flat, output_path=output_path)

    def plot_top_tags(self,
                      tags: list[str],
                      top_n: int = 10,
                      output_path: str = None):
        counter = Counter(tags)
        common = counter.most_common(top_n)
        tags, counts = zip(*common)
        font_path = r'C:\Windows\Fonts\msjh.ttc'
        my_font = fm.FontProperties(fname=font_path)
        if output_path:
            output_path += '.png'
            plt.figure(figsize=(10, 6))
            plt.bar(tags, counts, color='skyblue')
            plt.xticks(rotation=45, ha='right', fontproperties=my_font)
            plt.title(f'Top {top_n} Tags', fontproperties=my_font)
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] 標籤長條圖已儲存至 {output_path}")
            plt.show()
        else:
            plt.show()
        plt.close()

    def generate_wordcloud_tfidf(self,
                                 texts: list[str],
                                 font_path: str,
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
            plt.figure(figsize=(10, 8))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
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
                              font_path: str,
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
    years = ['106', '107', '108', '109', '110', '111', '113']
    #year = input('Please type the year of test:')
    for i in range(0,7):
        year = years[i]
        input_path = year + '.json'
        tags_output = year + 'tags'
        clouds_output = year + '.png'

        qa.tag_json_and_save(
            input_json=input_path,
            output_path=tags_output,
            top_k=3
        )

        qa.process_and_visualize(
            json_path=input_path,
            font_path=r'C:\Windows\Fonts\msjh.ttc',
            top_k=100,
            output_path=clouds_output
        )
