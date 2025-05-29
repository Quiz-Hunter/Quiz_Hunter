import json
import os
from glob import glob

# 設定資料夾路徑，請改成你放 JSON 檔案的資料夾
json_folder = "C:\\Users\\0524e\\OneDrive\\文件\\GitHub\\Quiz_Hunter\\Quiz_json"  # 替換成實際路徑，例如 "./year_jsons"

# 搜尋資料夾中所有 .json 檔案
json_files = sorted(glob(os.path.join(json_folder, "*.json")))

# 儲存所有題目的列表
all_questions = []

# 依序讀取每個 json 檔案並合併資料
for file_path in json_files:
    with open(file_path, encoding='utf-8') as f:
        year_data = json.load(f)
        all_questions.extend(year_data)

# 輸出合併後的 JSON 檔案
output_path = os.path.join(json_folder, "combined_106_to_113.json")
with open(output_path, "w", encoding='utf-8') as f:
    json.dump(all_questions, f, ensure_ascii=False, indent=2)

print(f"✅ 合併完成，輸出檔案路徑為：{output_path}")
