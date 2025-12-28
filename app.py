import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import requests
import json
import uuid
from datetime import datetime, timedelta
import altair as alt

# 页面设置
st.set_page_config(page_title="Vigil Dashboard - 深度分析", layout="wide", page_icon="📊")

# 数据存储路径 (与桌面端保持一致)
DATA_DIR = r"F:\Vigil\data"
LOG_FILE = os.path.join(DATA_DIR, "focus_history.csv")
REPORT_FILE = os.path.join(DATA_DIR, "reports.json")

# API 配置 (Kimi/Moonshot)
API_KEY = "sk-y8LGmh4LtgB3A2Dy5kRL9NZbXfdhWdLNpz8zT2v92Z2OTDv2"
API_URL = "https://api.moonshot.cn/v1/chat/completions"

# ================= 报告管理函数 =================
def load_reports():
    """加载历史报告"""
    if not os.path.exists(REPORT_FILE):
        return []
    try:
        with open(REPORT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []

def save_report(report_type, content, target_date=None):
    """保存新报告"""
    reports = load_reports()
    
    # 如果没有指定 target_date，默认使用当前日期的字符串 (YYYY-MM-DD)
    if not target_date:
        target_date = datetime.now().strftime("%Y-%m-%d")
        
    new_report = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_date": target_date, # 关联的日期
        "type": report_type, # "30min" or "daily"
        "content": content
    }
    reports.insert(0, new_report) # 插入到最前面
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    return new_report

def delete_report(report_id):
    """删除报告"""
    reports = load_reports()
    reports = [r for r in reports if r['id'] != report_id]
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)

def analyze_with_ai(prompt):
    """调用 Kimi API 进行分析"""
    try:
        messages = [
            {"role": "system", "content": "你是 Vigil 系统的智能效能分析师。请根据用户的数据提供专业、简练且富有同理心的分析报告。"},
            {"role": "user", "content": prompt}
        ]
        data = {"model": "moonshot-v1-8k", "messages": messages, "temperature": 0.7}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
        
        with st.spinner("🤖 Kimi 正在分析您的数据，请稍候..."):
            response = requests.post(API_URL, headers=headers, json=data, timeout=30)
            
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"分析失败: API 返回错误 {response.status_code}"
    except Exception as e:
        return f"分析请求出错: {str(e)}"

def load_data():
    """加载并预处理数据"""
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(LOG_FILE)
        if df.empty: return df
        
        # 转换时间戳
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['datetime'].dt.hour
        return df
    except:
        return pd.DataFrame()

def calculate_fatigue_cycle(df_today):
    """计算平均疲劳间隔 (分钟)"""
    drowsy_times = df_today[df_today['status'].isin(['Drowsy', 'Yawning'])]['datetime'].sort_values()
    if len(drowsy_times) < 2:
        return 0
    
    # 计算相邻疲劳点的时间差
    diffs = drowsy_times.diff().dt.total_seconds() / 60
    # 过滤掉太短的间隔（可能是连续记录），只看大于 10 分钟的间隔，认为是一次新的疲劳周期
    cycles = diffs[diffs > 10]
    
    if cycles.empty:
        return 0
    return round(cycles.mean(), 1)

def main():
    st.title("📊 Vigil 深度效能分析")
    
    if not os.path.exists(LOG_FILE):
        st.warning("暂无数据。请先启动桌面端悬浮窗 (desktop_pet.py) 进行监测。")
        if st.button("刷新"): st.rerun()
        return

    df = load_data()
    if df.empty:
        st.info("数据文件为空。")
        return

    # 侧边栏
    with st.sidebar:
        st.header("日期筛选")
        selected_date = st.date_input("选择日期", datetime.now())
        
        # 自动刷新功能
        auto_refresh = st.checkbox("⚡ 开启实时刷新 (每10秒)", value=False)
        
        if st.button("🔄 手动刷新数据"):
            st.rerun()
        
        if auto_refresh:
            time.sleep(10)
            st.rerun()
        
        st.divider()
        st.markdown("### 关于 Vigil")
        st.caption("数据由桌面悬浮窗自动采集。每 5 秒聚合一次。")

    # 过滤日期
    date_str = selected_date.strftime("%Y-%m-%d")
    df_day = df[df['date'] == date_str]

    if df_day.empty:
        st.info(f"{date_str} 暂无数据记录。")
        return

    # ================= 核心指标 =================
    total_samples = len(df_day)
    total_minutes = (total_samples * 5) // 60
    
    # 专注时间 (Focused)
    focused_samples = len(df_day[df_day['status'] == 'Focused'])
    focused_minutes = (focused_samples * 5) // 60
    
    # 疲劳次数
    drowsy_count = len(df_day[df_day['status'].isin(['Drowsy', 'Yawning'])])
    
    # 效能评分 (0-100) - 修复算法
    # 基础分30 + 专注加分 - 疲劳扣分
    focus_ratio = focused_samples / total_samples if total_samples > 0 else 0
    
    # 专注加分：0-50分（根据专注比例）
    focus_score = int(focus_ratio * 50)
    
    # 疲劳扣分：每小时疲劳次数 * 2分（降低惩罚）
    hours_monitored = max(1, total_minutes / 60)
    fatigue_penalty = int((drowsy_count / hours_monitored) * 2)
    
    # 基础分30 + 专注分 - 疲劳分，最低0最高100
    score = min(100, max(0, 30 + focus_score - fatigue_penalty))
    
    # 计算评分等级
    if score >= 80:
        grade = "优秀 🏆"
        delta_color = "normal"
    elif score >= 60:
        grade = "良好 👍"
        delta_color = "normal"
    elif score >= 40:
        grade = "及格 📊"
        delta_color = "off"
    else:
        grade = "需改进 ⚠️"
        delta_color = "inverse"
    
    score_display = f"{score} ({grade})"
    
    # 疲劳周期
    fatigue_cycle = calculate_fatigue_cycle(df_day)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("今日记录时长", f"{total_minutes} min")
    col2.metric("深度专注时长", f"{focused_minutes} min", help="状态为 Focused 的总时长")
    col3.metric("效能评分", score_display, delta=f"{score-60} vs 及格线", delta_color=delta_color)
    col4.metric("平均疲劳周期", f"{fatigue_cycle} min" if fatigue_cycle > 0 else "N/A", help="平均每隔多久出现一次疲劳")

    st.divider()

    # ================= 图表区域 =================
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("🔥 24小时专注热力分布")
        # 按小时聚合 EAR 均值
        hourly_stats = df_day.groupby('hour')['ear'].mean().reset_index()
        # 补全 0-23 小时
        all_hours = pd.DataFrame({'hour': range(24)})
        hourly_stats = pd.merge(all_hours, hourly_stats, on='hour', how='left').fillna(0)
        
        if not hourly_stats.empty:
            # 绘制柱状图
            chart = alt.Chart(hourly_stats).mark_bar().encode(
                x=alt.X('hour:O', title='时刻 (Hour)'),
                y=alt.Y('ear:Q', title='平均专注度 (EAR)'),
                color=alt.condition(
                    alt.datum.ear > 0.3,
                    alt.value('green'),  # 专注为绿
                    alt.value('lightgray')   # 普通为灰
                ),
                tooltip=['hour', 'ear']
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("今日暂无小时级数据。")

    with c2:
        st.subheader("📌 状态构成")
        status_counts = df_day['status'].value_counts()
        st.bar_chart(status_counts)

    st.subheader("📈 全天专注度趋势")
    # 降采样，避免图表过密，每分钟取一个点
    # 重要修正：使用 Pandas 的 to_datetime 确保索引是时间类型，否则 Altair 可能识别错误
    df_trend = df_day.copy()
    df_trend['datetime'] = pd.to_datetime(df_trend['datetime'])
    
    # 重新采样并填充缺失值，确保时间轴连续
    # 1. 设置索引
    df_trend = df_trend.set_index('datetime')
    # 2. 降采样 (每分钟)
    df_trend = df_trend.resample('1T')['ear'].mean().reset_index()
    
    # 修复时区问题：直接转换为简单的 HH:MM 字符串用于显示
    df_trend['time_str'] = df_trend['datetime'].dt.strftime('%H:%M')
    
    if not df_trend.empty and not df_trend['ear'].isna().all():
        # 清理 NaN 数据
        df_trend = df_trend.dropna(subset=['ear'])
        
        if not df_trend.empty:
            chart_line = alt.Chart(df_trend).mark_line(point=False).encode(
                # 使用 time_str 作为 X 轴，并按 datetime 排序
                x=alt.X('time_str', title='时间', sort=None), 
                y=alt.Y('ear:Q', scale=alt.Scale(domain=[0.15, 0.4]), title='专注度 (EAR)'), 
                tooltip=[
                    alt.Tooltip('time_str', title='时间'),
                    alt.Tooltip('ear', format='.3f', title='EAR')
                ]
            ).properties(height=300)
            
            # 添加阈值线 (分离为单独图表叠加)
            rule = alt.Chart(pd.DataFrame({'y': [0.22]})).mark_rule(color='red', strokeDash=[5, 5]).encode(y='y')
            st.altair_chart(chart_line + rule, use_container_width=True)
        else:
             st.info("有效数据点不足，趋势图暂未生成。")
    else:
        st.info("今日数据点较少，趋势图暂未生成。")
    
    st.caption("红虚线为疲劳阈值 (0.22)。低于此线表示可能处于疲劳状态。")
    
    st.divider()
    
    # ================= 智能报告区域 =================
    st.header("🤖 智能效能报告")
    
    r1, r2 = st.columns(2)
    
    with r1:
        st.subheader("⏱️ 30分钟即时洞察")
        st.markdown("分析最近 30 分钟的状态，提供即时调整建议。")
        
        if st.button("生成 30 分钟分析", key="btn_30min"):
            # 获取最近30分钟数据
            if not df.empty:
                last_time = df['datetime'].max()
                start_time = last_time - timedelta(minutes=30)
                df_recent = df[df['datetime'] >= start_time]
                
                if not df_recent.empty:
                    # 统计
                    rec_total = len(df_recent)
                    rec_focused = len(df_recent[df_recent['status'] == 'Focused'])
                    rec_drowsy = len(df_recent[df_recent['status'].isin(['Drowsy', 'Yawning'])])
                    rec_ear = df_recent['ear'].mean()
                    
                    prompt = (
                        f"请分析用户最近30分钟的精力状态：\n"
                        f"- 记录时长: {rec_total * 5 // 60} 分钟\n"
                        f"- 专注时长: {rec_focused * 5 // 60} 分钟\n"
                        f"- 疲劳/打哈欠次数: {rec_drowsy} 次\n"
                        f"- 平均专注度(EAR): {rec_ear:.3f}\n\n"
                        f"请给出简短的当前状态评估和接下来的行动建议（如继续保持或休息一下）。"
                    )
                    
                    analysis = analyze_with_ai(prompt)
                    
                    # 保存报告 (30分钟分析通常关联到今天)
                    save_report("30min", analysis, target_date=datetime.now().strftime("%Y-%m-%d"))
                    
                    st.success("分析完成！已保存至侧边栏历史记录。")
                    st.markdown(f"**Vigil 分析师**:\n\n{analysis}")
                    
                    # 刷新以更新侧边栏
                    time.sleep(2)
                    st.rerun()
                else:
                    st.warning("最近 30 分钟没有数据记录。")
            else:
                st.warning("没有数据可供分析。")

    with r2:
        st.subheader("📅 每日总结报告")
        st.markdown(f"基于 {date_str} 全天数据的深度复盘。")
        
        if st.button("生成今日总结", key="btn_daily"):
            if not df_day.empty:
                # 每日统计摘要
                daily_prompt = (
                    f"请为用户生成 {date_str} 的每日效能日报：\n"
                    f"- 总监测时长: {total_minutes} 分钟\n"
                    f"- 深度专注时长: {focused_minutes} 分钟\n"
                    f"- 效能评分: {score} ({grade})\n"
                    f"- 疲劳次数: {drowsy_count} 次\n"
                    f"- 平均疲劳周期: {fatigue_cycle} 分钟\n\n"
                    f"请总结用户今天的表现，指出精力高峰时段（如果有），并对明天的精力管理给出 3 条具体建议。"
                )
                
                daily_analysis = analyze_with_ai(daily_prompt)
                
                # 保存报告 (关键：关联到选中的 date_str)
                save_report("daily", daily_analysis, target_date=date_str)
                
                st.success("报告已生成！已保存至侧边栏历史记录。")
                st.markdown(f"**Vigil 分析师**:\n\n{daily_analysis}")
                
                # 刷新以更新侧边栏
                time.sleep(2)
                st.rerun()
            else:
                st.warning(f"{date_str} 暂无数据，无法生成报告。")
    
    st.divider()
    st.markdown("### 📜 历史报告")
    
    # 加载并显示历史报告
    reports = load_reports()
    
    # 过滤：只显示当前选中日期的报告
    # 优先使用 target_date 字段，如果没有则回退到 timestamp
    current_date_reports = []
    for r in reports:
        r_date = r.get('target_date', r['timestamp'][:10])
        if r_date == date_str:
            current_date_reports.append(r)
    
    if not current_date_reports:
        st.caption(f"{date_str} 暂无历史报告")
    else:
        for r in current_date_reports:
            # 标题显示时间 + 类型图标
            icon = "⏱️" if r['type'] == '30min' else "📅"
            title = f"{icon} {r['timestamp'][11:-3]}" # 只显示 HH:MM
            
            with st.expander(title):
                st.markdown(r['content'])
                if st.button("🗑️ 删除", key=f"del_{r['id']}"):
                    delete_report(r['id'])
                    st.rerun()

    st.divider()

if __name__ == "__main__":
    main()