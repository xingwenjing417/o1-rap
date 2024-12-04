import pandas as pd
import json

file_path_hard = "/Users/bytedance/Desktop/111 复杂题目 - Sheet1_hard.csv"
file_path_middle = "/Users/bytedance/Desktop/111 复杂题目 - Sheet1_middle.csv"
file_path_easy= "/Users/bytedance/Desktop/111 复杂题目 - Sheet1_easy.csv"
df_hard = pd.read_csv(file_path_hard)
df_middle = pd.read_csv(file_path_middle)
df_easy = pd.read_csv(file_path_easy)

if 'prompt' not in df_middle.columns:
    raise ValueError("表格中未找到 'prompt' 列。请检查表格内容。")
if 'prompt' not in df_easy.columns:
    raise ValueError("表格中未找到 'prompt' 列。请检查表格内容。")
if 'prompt' not in df_hard.columns:
    raise ValueError("表格中未找到 'prompt' 列。请检查表格内容。")

output_json = {
    "easy_code": {},
    "middle_code": {},
    "hard_code": {}
}
top_9_df_easy = df_easy.head(9)
random_12_df_middle = df_middle.head(36)
top_9_df_hard= df_hard.head(36)
#random_10_df = df.sample(n=10, random_state=42)
for idx, row in top_9_df_easy.iterrows():
    prompt_text = row.get("prompt", "").strip()
    if not prompt_text:
        continue  
    
    category = 'easy_code' #middle_code/hard_code
    if category:
        data_key = f"data{len(output_json[category]) + 1}"
        output_json[category][data_key] = {"query": prompt_text}
for idx, row in random_12_df_middle.iterrows():
    prompt_text = row.get("prompt", "")
    if not prompt_text:
        continue  
    
    category = 'middle_code' 
    if category:
        data_key = f"data{len(output_json[category]) + 1}"
        output_json[category][data_key] = {"query": prompt_text}
for idx, row in top_9_df_hard.iterrows():
    prompt_text = row.get("prompt", "").strip()
    if not prompt_text:
        continue  
    
    category = 'hard_code'
    if category:
        data_key = f"data{len(output_json[category]) + 1}"
        output_json[category][data_key] = {"query": prompt_text}

output_file_path = "/Users/bytedance/o1_rap/RAP/data/prontoqa/code_data_auto.json"
with open(output_file_path, "w", encoding="utf-8") as json_file:
    json.dump(output_json, json_file, ensure_ascii=False, indent=4)

print(f"处理完成，JSON文件已保存到：{output_file_path}")
