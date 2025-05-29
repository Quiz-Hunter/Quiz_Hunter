from transformers import pipeline
from difflib import SequenceMatcher
from dotenv import load_dotenv
import os
from google import genai
from langchain.llms.base import LLM
from typing import Optional, List, Any
from pydantic import PrivateAttr

class GoogleGeminiLLM(LLM):
    api_key: str
    model: str = "gemini-1.5-flash"
    _client: Any = PrivateAttr()

    @property
    def _llm_type(self) -> str:
        return "google_gemini"

    def __init__(self, **data):
        super().__init__(**data)
        genai.configure(api_key=self.api_key)
        self._client = genai.GenerativeModel(self.model)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self._client.generate_content(prompt)
        return response.text.strip()

    def answer_question(self, question: str, context: str) -> str:
        prompt = f"根據以下內容回答問題：\n內容：{context}\n問題：{question}"
        return self._call(prompt)

class DifficultyScorer:
    def __init__(self, question: dict):
        self.question = question
        self.context = question["content"]
        self.stem = self._extract_stem(self.context)

        self.models = {
            "small": pipeline("question-answering", model="uer/roberta-base-chinese-extractive-qa", tokenizer="uer/roberta-base-chinese-extractive-qa"),
            "medium": pipeline("question-answering", model="hfl/chinese-roberta-wwm-ext", tokenizer="hfl/chinese-roberta-wwm-ext"),
            "large": pipeline("question-answering", model="luhua/chinese_pretrain_mrc_roberta_wwm_ext_large", tokenizer="luhua/chinese_pretrain_mrc_roberta_wwm_ext_large"),
            "gold": self.google_llm_model()
        }

    def _extract_stem(self, content):
        return content.split("(A)")[0].strip()

    def _similar(self, a, b):
        return SequenceMatcher(None, a.strip(), b.strip()).ratio()

    def _is_correct(self, pred, answer, threshold=0.8):
        return self._similar(pred, answer) >= threshold

    def score(self):
    # 使用 Gemini 作為 gold standard
        gold = self.models["gold"].answer_question(self.stem, self.context)

        answers = {}
        for level in ["small", "medium", "large"]:
            try:
                pred = self.models[level](question=self.stem, context=self.context)["answer"]
                answers[level] = pred
            except Exception as e:
                print(f"[{level}] model error: {e}")
                answers[level] = ""

        # 比對準確與否
        small_correct  = self._is_correct(answers["small"], gold)
        medium_correct = self._is_correct(answers["medium"], gold)
        large_correct  = self._is_correct(answers["large"], gold)

        # 四級分類邏輯
        if small_correct and medium_correct and large_correct:
            difficulty = "簡單"
        elif medium_correct and large_correct:
            difficulty = "中低"
        elif large_correct:
            difficulty = "中"
        else:
            difficulty = "困難"

        return difficulty, gold, answers


    def google_llm_model(self):
        load_dotenv("config.env") 
        api_key = os.getenv("GOOGLE_API_KEY")
        return GoogleGeminiLLM(api_key=api_key)

if __name__ == "__main__":
    query = {
        "content": "下列哪一種物質與適當的催化劑共熱，可得到氧氣？ (A) 水 (B) 氯酸鉀 (C) 碳酸鈣 (D) 硫酸鉀 (E) 碳酸氫鈉"
    }

    scorer = DifficultyScorer(query)
    level, gold, answers = scorer.score()
    print(f"原題幹：{query['content']}")
    print(f"🔑 正解（Gemini）：{gold}")
    print(f"🧠 難度：{level}")
    for k, v in answers.items():
        print(f"🔍 {k}：{v}")
