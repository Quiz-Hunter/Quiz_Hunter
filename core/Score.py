import os
import re
from typing import Optional, List, Any

import google.generativeai as genai
from dotenv import load_dotenv
from transformers import pipeline
from langchain.llms.base import LLM
from pydantic import PrivateAttr


# --- Google Gemini LLM 包裝 ---
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
        self._client = genai.GenerativeModel(model_name=self.model)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self._client.generate_content(prompt)
        return response.text.strip()

    def answer_question(self, question: str, context: str, role: Optional[str] = None) -> str:
        if role:
            role_instruction = {
                "small": "你只具備國中程度的知識與理解力，容易混淆名詞或依直覺猜測答案，常常忽略關鍵細節。請依據片段記憶與語感作答，避免過度推理。",
                "medium": "你只具備高一自然科學生的程度，知道一些常見的化學名詞與基本反應，會試著根據題目的字面意思作答，但容易被關鍵詞或看起來熟悉的選項誤導。請依據你記得的知識和直覺作答，並避免過度推理或使用反應式。",
                "large": "你是高中自然科普通的學生，具備邏輯與整合推理能力，能掌握題幹細節並作出合理分析。"
            }.get(role, "請根據你能力回答")

            prompt = f"""
            你是一位 {role} 中文語言模型：
            {role_instruction}

            請閱讀以下題目後根據能力進行回答，並用以下格式回答：

            選項：X
            理由：...

            題目：{question}
            {context}
            """
        else:
            prompt = f"""
            根據以下內容回答問題，請用以下格式回答：

            選項：X
            理由：...

            題目：{question}
            {context}
            """
        return self._call(prompt)

    def judge_answer(self, question: str, context: str, candidate: str, reason: str) -> str:
        # 第一層：判斷選項是否正確
        prompt_1 = f"""
        你是一位自然科題目批改助理，請判斷以下學生的選項是否正確。

        題目：{question}
        {context}
        學生作答：{candidate}

        請回覆：「正確」或「錯誤」。
        """
        result_1 = self._call(prompt_1)

        if "錯誤" in result_1:
            return "錯誤"

        # 第二層：判斷理由是否展現出完整自信理解
        prompt_2 = f"""
        你是一位自然科老師，請判斷以下學生對於題目的理解是否清晰、具信心、且無模糊推測。

        題目：{question}
        {context}
        作答理由：{reason}

        若學生的理由中有以下語句，表示其理解不完整、不具信心，請回覆「理解不完全」：
        「感覺」「好像」「應該是」「不太懂」「不確定」「我猜」「我沒辦法判斷」「我無法理解」「這我不清楚」「不太會判斷」「這題有點難」

        若理由清晰完整，請回覆「理解完整」。
        """
        result_2 = self._call(prompt_2)

        if "理解不完全" in result_2:
            return "錯誤"

        return "正確"


# --- DifficultyScorer 評估器 ---
class DifficultyScorer:
    def __init__(self, question: dict):
        self.question = question
        self.context = self._normalize_context(question["content"])
        self.stem = self._extract_stem(self.context)

        self.models = {
            "gold": self.google_llm_model()
        }

    def _extract_stem(self, content):
        return content.split("(A)")[0].strip()

    def _normalize_context(self, content):
        options = re.findall(r"\([A-E]\)[^\(\)]+", content)
        context = self._extract_stem(content) + "\n選項：\n" + "\n".join(options)
        return context

    def google_llm_model(self):
        load_dotenv("config.env")
        api_key = os.getenv("GOOGLE_API_KEY")
        return GoogleGeminiLLM(api_key=api_key)

    def gemini_rating(self) -> int:
        prompt = f"""
        請你扮演一位專業的高中自然科命題老師，請幫以下題目評估難易程度，並回覆整數難度分數（1~5顆星）：
        1 顆星表示非常簡單，5 顆星表示非常困難。

        題目內容：
        {self.context}

        請回覆格式為：「難度：X 顆星」。
        """
        reply = self.models["gold"]._call(prompt)
        match = re.search(r"難度[:：]?\s*(\d)", reply)
        return int(match.group(1)) if match else 3

    def score(self):
        gold = self.models["gold"].answer_question(self.stem, self.context)

        answers = {}
        for level in ["small", "medium", "large"]:
            try:
                pred = self.models["gold"].answer_question(self.stem, self.context, role=level)
                match = re.search(r"選項[:：]?\s*([A-E])", pred)
                choice = match.group(1) if match else "未知"
                reason_match = re.search(r"理由[:：]?\s*(.*)", pred)
                reason = reason_match.group(1).strip() if reason_match else "（無理由）"
                answers[level] = {"choice": choice, "reason": reason}
            except Exception as e:
                print(f"[{level}] 模擬失敗: {e}")
                answers[level] = {"choice": "", "reason": "錯誤"}

        correctness = {}
        for level in ["small", "medium", "large"]:
            a = answers[level]
            judgment = self.models["gold"].judge_answer(self.stem, self.context, a["choice"], a["reason"])
            correctness[level] = "正確" in judgment

        if correctness["small"] and correctness["medium"] and correctness["large"]:
            star_auto = 2
        elif correctness["medium"] and correctness["large"]:
            star_auto = 3
        elif correctness["large"]:
            star_auto = 4
        else:
            star_auto = 5

        star_gemini = self.gemini_rating()
        final_star = round((star_auto * 0.4 + star_gemini * 0.6))

        return final_star, gold, answers, correctness, star_auto, star_gemini


# --- 測試程式 ---
if __name__ == "__main__":
    query = {
        "content": "在某密閉容器中，加入過量的鐵粉並通入適量的氯氣，發現反應生成紅棕色的固體，並伴隨放熱現象。下列關於此反應的敘述，何者正確？ (A) 此反應為還原反應，生成物為 FeCl (B) 此反應吸熱，表示生成物比反應物穩定 (C) 此反應屬於氧化還原反應，生成物為 FeCl₃ (D) 氯氣作為還原劑，將鐵還原為 Fe²⁺ (E) 若容器內壓力上升，代表反應消耗氣體體積小於生成氣體"
    }

    scorer = DifficultyScorer(query)
    stars, gold, answers, correctness, auto, gem = scorer.score()
    print(f"\U0001f511 正解（Gemini）：{gold}")
    print(f"\U0001f9e0 難度（自動答題評估）：{auto} 星")
    print(f"\U0001f4ca 難度（Gemini語意評估）：{gem} 星")
    print(f"⭐️ 綜合難度評等：{stars} 星")
    for k in ["small", "medium", "large"]:
        a = answers[k]
        mark = "✅" if correctness[k] else "❌"
        print(f"\U0001f50d {k}：{a['choice']} {mark} 理由：{a['reason']}")