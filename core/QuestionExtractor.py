import re, json
from pathlib import Path
import pymupdf4llm

class QuestionExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = Path(pdf_path)
        self.year = "unknown"
        self.subject = "unknown"

    def clean_md_text(self, md_text):
        text = md_text
        text = re.sub(r'-+\s*\d+\s*-+\n', '', text)
        text = re.sub(r'\[image:.*?\]', '', text)
        text = re.sub(r'^\s*(圖|表)\s*\d+.*$', '', text, flags=re.M)
        text = re.sub(r'The following table:', '', text)
        text = re.sub(r'\n\s*\n+', '\n', text)
        return text.strip()


    def extract_exam_info(self, md_text):
         # 全形數字轉半形
        md_text = md_text.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
        # 搜尋頁尾資訊
        pattern = r'第\s*\d+\s*頁\s*(\d{3,4})年學測\s*共\s*\d+\s*頁\s*([^\s\d]+)考科'
        match = re.search(pattern, md_text)
        if match:
            year_match = match.group(1)
            subject_match = match.group(2)
            self.year = year_match 
            self.subject = subject_match
        else:
            self.year = "unknown"
            self.subject =  "unknown"

    def extract_groups(self, md_text):
        group_pattern = re.compile(r'^\s*(\d+)\s*-\s*(\d+)\s*為題組\s*\n([\s\S]*?)(?=\n\s*\d+\.)', re.M)
        groups = {}
        for match in group_pattern.finditer(md_text):
            start, end, context = int(match.group(1)), int(match.group(2)), match.group(3).strip()
            groups[f"{start}-{end}"] = {"start": start, "end": end, "context": context}
        return groups

    def extract_questions(self, md_text, groups):
        questions = []
        question_pattern = re.compile(r'^\s*(\d+)\.\s*([\s\S]*?)(?=\n\s*\d+\.|\Z)', re.M)
        option_pattern = re.compile(
            r'\(([A-Za-z甲乙丙丁戊己庚辛壬癸])\)\s*(.*?)(?=\s*(?:\([A-Za-z甲乙丙丁戊己庚辛壬癸]\)|\n\s*\d+\.|\n\s*\d+\s*-\s*\d+\s*為題組|\Z))',
            re.S)

        for match in question_pattern.finditer(md_text):
            q_num, content = int(match.group(1)), match.group(2).strip()
            if q_num == 0:
                continue
            stem = content
            options = {}
            first_option_match = re.search(r'\([A-Za-z甲乙丙丁戊己庚辛壬癸]\)', content)
            if first_option_match:
                stem = content[:first_option_match.start()].strip()
                options_text = content[first_option_match.start():].strip()
                found_options = option_pattern.findall(options_text)
                for label, text in found_options:
                    cleaned_text = ' '.join(text.split())
                    options[label] = cleaned_text

            group_id = next((gid for gid, g in groups.items() if g["start"] <= q_num <= g["end"]), None)
            group_context = groups[group_id]["context"] if group_id else ""

            questions.append({
                "id": q_num,
                "year": self.year,
                "subject": self.subject,
                "group_id": group_id,
                "group_context": ' '.join(group_context.split()),
                "stem": ' '.join(stem.split()),
                "options": options
            })

        questions.sort(key=lambda x: x["id"])
        return questions

    def process_pdf(self, output_json_path):
        md_content = pymupdf4llm.to_markdown(str(self.pdf_path))

        # 提取考試年度與科目
        self.extract_exam_info(md_content)

        clean_md = self.clean_md_text(md_content)
        groups = self.extract_groups(clean_md)
        questions = self.extract_questions(clean_md, groups)

        Path(output_json_path).write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ 提取完成 年度: {self.year}, 科目: {self.subject}, 共 {len(questions)} 題，儲存於 {output_json_path}")


