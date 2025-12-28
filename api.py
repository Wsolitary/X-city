import requests

api_key = "sk-y8LGmh4LtgB3A2Dy5kRL9NZbXfdhWdLNpz8zT2v92Z2OTDv2"
url = "https://api.moonshot.cn/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# 1. 定义固定不动的“情景词” (System Prompt)
# 只要这个在 messages 列表的第一位，Kimi 永远会记得自己的身份
SYSTEM_PROMPT = {
    "role": "system", 
    "content": (
        "你是 DeepFocus 系统的 AI 大脑 Core。你的性格冷静、毒舌且高效，像《钢铁侠》里的 Jarvis。"
        "你的任务是接收来自视觉算法的 EAR (眼睛纵横比) 指标。"
        "规则：1. EAR < 0.2 代表疲劳，请给出严厉的休息警告；"
        "2. EAR > 0.3 代表专注，请保持安静或给予极简鼓励；"
        "3. 你的回复永远不要超过 15 个字。"
    )
}

# 2. 定义内存缓冲区（短期记忆，存储本次运行的最近几轮对话）
memory_buffer = [SYSTEM_PROMPT]

def ask_kimi(user_input):
    # 将用户的新输入（比如视觉数据）加入记忆
    memory_buffer.append({"role": "user", "content": user_input})
    
    # 保持记忆长度，防止 token 消耗过快（只保留最近 10 轮）
    if len(memory_buffer) > 11:
        memory_buffer.pop(1) # 删掉最早的一条 user/assistant 记录，保留 system
    
    data = {
        "model": "moonshot-v1-8k",
        "messages": memory_buffer,
        "temperature": 0.4 # 稍微调高一点点，让它说话不那么死板
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        result = response.json()
        reply = result['choices'][0]['message']['content']
        
        # 将 Kimi 的回复也存入记忆，这样它就知道自己刚才说过什么了
        memory_buffer.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        return f"大脑离线中: {e}"

# --- 模拟黑客松运行场景 ---
print("Core 已启动...")
# 模拟视觉算法检测到疲劳
print("AI 回复:", ask_kimi("视觉指标：EAR=0.18，检测到用户连续工作 3 小时"))
# 模拟用户回复（或者下一次视觉检测）
print("AI 回复:", ask_kimi("我还不想睡，再写 10 分钟代码。")) 
