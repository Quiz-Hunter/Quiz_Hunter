from transformers import pipeline
from difflib import SequenceMatcher


class DifficultyScorer:
    def __init__(self, question: dict):
        """
        question: dict 來自 BM25HNSWRetriever.search() 的單筆結果
        """
        self.question = question
        self.context = question["content"]
        self.stem = self._extract_stem(question["content"])
        self.models = {
            "small": pipeline("question-answering", model="uer/roberta-base-chinese-extractive-qa", tokenizer="uer/roberta-base-chinese-extractive-qa"),
            "medium": pipeline("question-answering", model="hfl/chinese-roberta-wwm-ext", tokenizer="hfl/chinese-roberta-wwm-ext"),
            "large": pipeline("question-answering", model="luhua/chinese_pretrain_mrc_roberta_wwm_ext_large", tokenizer="luhua/chinese_pretrain_mrc_roberta_wwm_ext_large"),
        }

    def _extract_stem(self, content):
        """
        嘗試從 content 中取出題幹（預設為選項之前的第一句）
        """
        return content.split("(A)")[0].strip()

    def _similar(self, a, b):
        return SequenceMatcher(None, a.strip(), b.strip()).ratio()

    def _is_correct(self, pred, answer, threshold=0.8):
        return self._similar(pred, answer) >= threshold

    def score(self):
        gold = self.models["large"](question=self.stem, context=self.context)["answer"]

        answers = {}
        for level in ["small", "medium"]:
            try:
                pred = self.models[level](question=self.stem, context=self.context)["answer"]
                answers[level] = pred
            except:
                answers[level] = ""

        if self._is_correct(answers["small"], gold):
            return "簡單", gold, answers
        elif self._is_correct(answers["medium"], gold):
            return "適中", gold, answers
        else:
            return "困難", gold, answers

if __name__ == "__main__":


    query = {"content":"下列哪一種物質與適當的催化劑共熱，可得到氧氣？ (A) 水 (B) 氯酸鉀 (C) 碳酸鈣 (D) 硫酸鉀 (E) 碳酸氫鈉"}

    scorer = DifficultyScorer(query)
    level, gold, answers = scorer.score()
    print(f"原題幹：{query}")
    print(f"🔑 正解（large model）：{gold}")
    print(f"🧠 難度：{level}")
    print(f"🔍 small：{answers['small']}")
    print(f"🔍 medium：{answers['medium']}")