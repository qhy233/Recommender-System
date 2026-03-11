import json
import csv
import time
from openai import OpenAI
import os

# --- 1. 配置区域 ---
API_KEY = "sk-72c5603f44a74ab5b8831a42e6ec5387"  # ⚠️ 替换你的Key
BASE_URL = "https://api.deepseek.com"

# 我们要测试的题目（这里先放两道题做测试）
# 修改 simulation.py 里的这一段

QUESTIONS = [
    "1. 我觉得比平时容易紧张和着急 (Anxiety/Nervousness)",
    "2. 我无缘无故地感到害怕 (Fear without reason)",
    "3. 我容易心里烦乱或觉得惊恐 (Panic)",
    "4. 我觉得我可能将要发疯 (Feeling of going crazy)",
    "5. 我觉得一切都很好 (Everything is fine) [注意：这题是反向计分题，健康人应该选高分]"
]

# --- 2. 初始化 ---
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ✨✨✨ 修改开始 ✨✨✨
# 自动获取 simulation.py 所在的文件夹路径
base_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接出 agents.json 的绝对路径
json_path = os.path.join(base_dir, 'agents.json')

# 读取人设文件 (注意这里换成了 json_path)
with open(json_path, 'r', encoding='utf-8') as f:
    agents = json.load(f)

# 准备写入 CSV 文件
csv_file = open('results.csv', 'w', newline='', encoding='utf-8-sig') # utf-8-sig 防止Excel乱码
writer = csv.writer(csv_file)
# 写入表头：ID, 组别, Q1答案, Q2答案...
header = ['Agent_ID', 'Group'] + [f'Q{i+1}' for i in range(len(QUESTIONS))]
writer.writerow(header)

print(f"🎬 仿真开始！共有 {len(agents)} 个 Agent，每人做 {len(QUESTIONS)} 道题。")
print("-" * 50)

# --- 3. 核心循环 (你的项目核心) ---
for agent in agents:
    print(f"🤖 正在测试 Agent {agent['id']} ({agent['group']})...")
    
    agent_answers = [agent['id'], agent['group']] # 先记下ID和组别
    
    for q in QUESTIONS:
        # 构建 Prompt
        system_prompt = f"""
        {agent['persona']}
        
        【任务】
        请完成下面的心理测试题。
        题目：{q}
        选项：1-完全不符，2-有点不符，3-说不清，4-有点符合，5-完全符合。
        
        【要求】
        只输出一个数字（1-5），不要包含任何标点或文字解释。
        """
        
        try:
            # 调用 API
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "请打分："}
                ],
                temperature=0.1 # 让它稍微稳定点
            )
            answer = response.choices[0].message.content.strip()
            print(f"   📝 题目: {q[:10]}... -> 得分: {answer}")
            agent_answers.append(answer)
            
        except Exception as e:
            print(f"   ❌ 出错: {e}")
            agent_answers.append("Error")
        
        # 稍微停顿一下，防止太快（其实DeepSeek很快，不加也行，但加了稳妥）
        # time.sleep(0.5) 

    # 把这个人的所有答案写入 CSV
    writer.writerow(agent_answers)
    print(f"✅ Agent {agent['id']} 测试完成！\n")

# --- 4. 收尾 ---
csv_file.close()
print("-" * 50)
print("🎉 所有仿真结束！结果已保存到 results.csv")