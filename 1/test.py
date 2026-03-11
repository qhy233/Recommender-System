from openai import OpenAI

# 1. 配置连接信息
client = OpenAI(
    # ⚠️ 务必替换为你刚刚生成的 Key (保留引号)
    api_key="sk-72c5603f44a74ab5b8831a42e6ec5387", 
    base_url="https://api.deepseek.com"  # DeepSeek 的接口地址
)

print("正在连接 DeepSeek 服务器...")

# 2. 发送请求
try:
    response = client.chat.completions.create(
        model="deepseek-chat",  # 指定模型版本
        messages=[
            {"role": "system", "content": "你是一个心理学专家。"},
            {"role": "user", "content": "你好，请用一句话告诉我什么是习得性无助？"}
        ],
        stream=False
    )
    
    # 3. 打印结果
    print("连接成功！AI 回复如下：")
    print("-" * 30)
    print(response.choices[0].message.content)
    print("-" * 30)

except Exception as e:
    print(f"出错了：{e}")