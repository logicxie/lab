import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import json
import os
import re
import uuid
import calendar
from datetime import datetime, timedelta

# 1. 设置页面适应手机屏幕，并默认收起侧边栏（如果有的话）
st.set_page_config(page_title="细胞实验管理", layout="centered", initial_sidebar_state="collapsed")

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;} /* 隐藏右上角菜单 */
    footer {visibility: hidden;}    /* 隐藏底部水印 */
    header {visibility: hidden;}    /* 隐藏顶部的彩色装饰条 */
    
    /* ========================================================
       终极日历自适应补丁 (兼容新老版本 Streamlit, 强杀移动端换行)
       ======================================================== */
       
    /* ----- 1. 强制 3列的月份导航头在移动端不换行 ----- */
    div[data-testid="stHorizontalBlock"]:has(.cal-month-header),
    div[data-testid="stColumns"]:has(.cal-month-header) {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.cal-month-header) > div,
    div[data-testid="stColumns"]:has(.cal-month-header) > div {
        min-width: 0 !important; /* 解除宽度霸权 */
    }
    div[data-testid="stHorizontalBlock"]:has(.cal-month-header) > div:nth-child(1),
    div[data-testid="stHorizontalBlock"]:has(.cal-month-header) > div:nth-child(3),
    div[data-testid="stColumns"]:has(.cal-month-header) > div:nth-child(1),
    div[data-testid="stColumns"]:has(.cal-month-header) > div:nth-child(3) {
        flex: 1 1 25% !important;
        width: 25% !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.cal-month-header) > div:nth-child(2),
    div[data-testid="stColumns"]:has(.cal-month-header) > div:nth-child(2) {
        flex: 1 1 50% !important;
        width: 50% !important;
    }
    .cal-month-header {
        font-size: clamp(1rem, 4vw, 1.5rem) !important; 
        white-space: nowrap !important;
    }

    /* ----- 2. 强制 7列的日历表头和日期主体在移动端绝对不换行 ----- */
    div[data-testid="stHorizontalBlock"]:has(> div:nth-child(7)),
    div[data-testid="stColumns"]:has(> div:nth-child(7)) {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: 2px !important; 
    }
    
    div[data-testid="stHorizontalBlock"]:has(> div:nth-child(7)) > div,
    div[data-testid="stColumns"]:has(> div:nth-child(7)) > div {
        min-width: 0 !important; 
        flex: 1 1 calc(100% / 7) !important; 
        width: calc(100% / 7) !important;
        padding: 0 !important; 
    }

    /* ----- 3. 日期按钮：强制正圆形，完全自适应 ----- */
    div[data-testid="stHorizontalBlock"]:has(> div:nth-child(7)) button,
    div[data-testid="stColumns"]:has(> div:nth-child(7)) button {
        width: 100% !important;
        aspect-ratio: 1 / 1 !important; /* 关键：确保宽高完全 1:1 */
        min-height: 0 !important;       
        height: auto !important;
        border-radius: 50% !important;  
        padding: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        box-sizing: border-box !important;
    }

    /* 调整圆圈内部文字大小，开启 pre-wrap 让标记图案正常换行 */
    div[data-testid="stHorizontalBlock"]:has(> div:nth-child(7)) button p,
    div[data-testid="stColumns"]:has(> div:nth-child(7)) button p,
    div[data-testid="stHorizontalBlock"]:has(> div:nth-child(7)) button div,
    div[data-testid="stColumns"]:has(> div:nth-child(7)) button div {
        font-size: clamp(0.6rem, 2.2vw, 1.1rem) !important; 
        line-height: 1.1 !important;
        margin: 0 !important;
        white-space: pre-wrap !important; 
        text-align: center !important;
    }
    
    /* 星期表头样式居中对齐 */
    .cal-weekday {
        font-size: clamp(0.7rem, 3vw, 1rem) !important;
        text-align: center !important;
        padding-bottom: 5px !important;
    }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

AREA_MAP = {
    "10cm 培养皿": 55.0, "6cm 培养皿": 21.0, "T75 培养瓶": 75.0,
    "T25 培养瓶": 25.0, "6孔板 (单孔)": 9.6, "12孔板 (单孔)": 3.8,
    "24孔板 (单孔)": 1.9, "96孔板 (单孔)": 0.32,
    "6孔板 (整板)": 9.6 * 6, "12孔板 (整板)": 3.8 * 12,
    "24孔板 (整板)": 1.9 * 24, "96孔板 (整板)": 0.32 * 96
}

# ==========================================
# 1. 核心数学模型
# ==========================================
def logistic_model(X, r):
    N0, t = X
    K = 100.0
    N0 = np.clip(N0, 0.1, 99.9)
    return K / (1 + (K/N0 - 1) * np.exp(-r * t))

def calculate_inverse_N0(Nt, t, r):
    K = 100.0
    Nt = min(Nt, 99.9)
    return K / (((K/Nt) - 1) * np.exp(r * t) + 1)

from sqlalchemy import text

# ==========================================
# 2. 数据库持久化与初始化 (Supabase PostgreSQL)
# ==========================================

conn = st.connection("supabase", type="sql")

def _init_db():
    with conn.session as s:
        # 创建 key-value 数据表
        s.execute(text("CREATE TABLE IF NOT EXISTS store (key TEXT PRIMARY KEY, data TEXT)"))
        s.commit()
    st.cache_data.clear()

_init_db()

def _load_data(key, default_val):
    try:
        df_res = conn.query("SELECT data FROM store WHERE key = :key", params={"key": key}, ttl="10m")
        if not df_res.empty:
            return json.loads(df_res.iloc[0]["data"])
    except Exception:
        return default_val
    return default_val

def _save_data(key, data):
    with conn.session as s:
        # 存入数据库前将对象转为JSON字符串, 使用 PG 的 ON CONFLICT DO UPDATE 实现 Upsert
        s.execute(
            text("INSERT INTO store (key, data) VALUES (:key, :data) ON CONFLICT (key) DO UPDATE SET data = EXCLUDED.data"),
            {"key": key, "data": json.dumps(data, ensure_ascii=False)}
        )
        s.commit()
    st.cache_data.clear()

def load_cell_db():
    db_dict = _load_data("cell_db", {})
    if db_dict:
        for k, v in db_dict.items():
            v["data"] = pd.DataFrame(v["data"])
            if "pre_passage_records" in v:
                v["pre_passage_records"] = pd.DataFrame(v["pre_passage_records"])
            else:
                v["pre_passage_records"] = pd.DataFrame(columns=["传代前容器", "传前密度 (%)", "传代比例 (%)", "传代后容器", "日期"])
        return db_dict
    return {
        "示例细胞_10%FBS": {
            "data": pd.DataFrame({"N0": [33.0], "t": [24.0], "Nt": [80.0], "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]}),
            "r": 0.08,
            "pre_passage_records": pd.DataFrame(columns=["传代前容器", "传前密度 (%)", "传代比例 (%)", "传代后容器", "日期"])
        }
    }

def save_cell_db():
    db_dict = {}
    for k, v in st.session_state.cell_db.items():
        db_dict[k] = {
            "data": v["data"].to_dict(orient="records"), 
            "r": v["r"],
            "pre_passage_records": v.get("pre_passage_records", pd.DataFrame()).to_dict(orient="records")
        }
    _save_data("cell_db", db_dict)

# 初始化 Session State
if 'cell_db' not in st.session_state: st.session_state.cell_db = load_cell_db()
if 'schedule' not in st.session_state: st.session_state.schedule = _load_data("schedule", [])
if 'inventory' not in st.session_state: st.session_state.inventory = _load_data("inventory", [])
if 'journal' not in st.session_state: st.session_state.journal = _load_data("journal", [])
if 'memo' not in st.session_state: st.session_state.memo = _load_data("memo", [])
if 'sops' not in st.session_state: st.session_state.sops = _load_data("sops", [])
if 'animals' not in st.session_state: st.session_state.animals = _load_data("animals", [])
if 'bioinfo' not in st.session_state: st.session_state.bioinfo = _load_data("bioinfo", [])
if 'results' not in st.session_state: st.session_state.results = _load_data("results", [])

# ==========================================
# 工具函数 (替代 _save_json)
# ==========================================
def _save_schedule():
    _save_data("schedule", st.session_state.schedule)

def _save_inventory():
    _save_data("inventory", st.session_state.inventory)

def _save_journal():
    _save_data("journal", st.session_state.journal)

def _save_memo():
    _save_data("memo", st.session_state.memo)

def _save_sops():
    _save_data("sops", st.session_state.sops)

def _save_animals():
    _save_data("animals", st.session_state.animals)

def _save_bioinfo():
    _save_data("bioinfo", st.session_state.bioinfo)

def _save_results():
    _save_data("results", st.session_state.results)

# ==========================================
# 3. 共享 UI 组件 (日历引擎保持不变)
# ==========================================
def interactive_calendar(calendar_type="all"):
    key_prefix = calendar_type
    if f"cal_current_month_{key_prefix}" not in st.session_state:
        st.session_state[f"cal_current_month_{key_prefix}"] = datetime.now().date().replace(day=1)
    if "cal_selected_date" not in st.session_state:
        st.session_state["cal_selected_date"] = datetime.now().date()
        
    current_month = st.session_state[f"cal_current_month_{key_prefix}"]
    
    col_prev, col_month, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("◀ 上个月", key=f"prev_{key_prefix}", use_container_width=True):
            prev_m = current_month.month - 1
            y = current_month.year if prev_m > 0 else current_month.year - 1
            st.session_state[f"cal_current_month_{key_prefix}"] = current_month.replace(year=y, month=prev_m if prev_m > 0 else 12)
            st.rerun()
    with col_month:
        st.markdown(f"<h4 class='cal-month-header' style='text-align: center; margin:0;'>{current_month.strftime('%Y年 %m月')}</h4>", unsafe_allow_html=True)
    with col_next:
        if st.button("下个月 ▶", key=f"next_{key_prefix}", use_container_width=True):
            next_m = current_month.month + 1
            y = current_month.year if next_m <= 12 else current_month.year + 1
            st.session_state[f"cal_current_month_{key_prefix}"] = current_month.replace(year=y, month=next_m if next_m <= 12 else 1)
            st.rerun()
            
    active_memo_dates = {pd.to_datetime(m["time"]).date() for m in st.session_state.memo if m.get("status", "pending") != "completed"} if calendar_type in ["memo", "all"] else set()
    done_memo_dates = {pd.to_datetime(m["time"]).date() for m in st.session_state.memo if m.get("status", "pending") == "completed"} if calendar_type in ["memo", "all"] else set()
    
    active_journal_dates = {pd.to_datetime(j["datetime"]).date() for j in st.session_state.journal if j.get("status", "⏳ 待执行") != "✅ 已完成"} if calendar_type in ["journal", "all"] else set()
    done_journal_dates = {pd.to_datetime(j["datetime"]).date() for j in st.session_state.journal if j.get("status", "⏳ 待执行") == "✅ 已完成"} if calendar_type in ["journal", "all"] else set()
    
    active_sched_dates = {pd.to_datetime(s["obs_time"]).date() for s in st.session_state.schedule if s.get("status", "⏳ 待执行") != "✅ 已完成"} if calendar_type in ["schedule", "all"] else set()
    done_sched_dates = {pd.to_datetime(s["obs_time"]).date() for s in st.session_state.schedule if s.get("status", "⏳ 待执行") == "✅ 已完成"} if calendar_type in ["schedule", "all"] else set()
    
    weekdays = ["一", "二", "三", "四", "五", "六", "日"]
    cols = st.columns(7)
    for i, wd in enumerate(weekdays):
        cols[i].markdown(f"**<div class='cal-weekday' style='text-align:center;'>{wd}</div>**", unsafe_allow_html=True)
        
    cal_m = calendar.Calendar(firstweekday=0)
    month_days = cal_m.monthdatescalendar(current_month.year, current_month.month)
    
    for week in month_days:
        cols = st.columns(7)
        for i, day in enumerate(week):
            if day.month != current_month.month:
                with cols[i]: st.write("") 
                continue
            
            with cols[i]:
                marks = []
                if day in active_memo_dates: marks.append("🔵")
                if day in active_journal_dates: marks.append("🟢")
                if day in active_sched_dates: marks.append("🟠")
                if day in done_sched_dates or day in done_journal_dates or day in done_memo_dates: marks.append("⚪")
                
                label = f"{day.day}\n{''.join(marks)}" if marks else str(day.day)
                is_selected = (day == st.session_state.cal_selected_date)
                
                if st.button(label, key=f"btn_{key_prefix}_{day}", use_container_width=True, type="primary" if is_selected else "secondary"):
                    st.session_state.cal_selected_date = day
                    st.rerun()

# ==========================================
# 4. 各功能模块渲染函数
# ==========================================

def render_home():
    st.title("🏠 实验室中央控制台")
    
    # 新增：全局核心指标概览
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("待办备忘录", len(st.session_state.memo))
    m2.metric("进行中实验/传代", len([j for j in st.session_state.journal if '待执行' in j['status']]) + len(st.session_state.schedule))
    m3.metric("活体动物队列", len([a for a in st.session_state.animals if a['status'] == '造模/观察中']))
    m4.metric("生信分析项目", len(st.session_state.bioinfo))
    st.divider()

    st.markdown("📅 **实验日程全量总览** (点击日期查看)")
    st.caption("标记提醒分类指示: 🔵 备忘录 | 🟢 实验日志 | 🟠 待办传代 | ⚪ 已完成传代")
    interactive_calendar("all")
    st.divider()
    
    selected_d = st.session_state.get("cal_selected_date", datetime.now().date())
    st.markdown(f"### 📌 {selected_d.strftime('%Y年%m月%d日')} 事务明细")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🔵 个人备忘")
        memos = sorted([m for m in st.session_state.memo if pd.to_datetime(m["time"]).date() == selected_d], key=lambda x: x['time'])
        for m in memos:
            if m.get('status') == "completed":
                st.success(f"**{pd.to_datetime(m['time']).strftime('%H:%M')} | {m['title']}** (已归档)\n\n{m['content']}")
            else:
                st.info(f"**{pd.to_datetime(m['time']).strftime('%H:%M')} | {m['title']}**\n\n{m['content']}")
        if not memos: st.caption("无备忘安排")
        
    with c2:
        st.markdown("#### 🟢 实验日志")
        jours = sorted([j for j in st.session_state.journal if pd.to_datetime(j["datetime"]).date() == selected_d], key=lambda x: x['datetime'])
        for j in jours: 
            if j['status'] == "⏳ 待执行": st.warning(f"⏳ **{pd.to_datetime(j['datetime']).strftime('%H:%M')} | {j['title']}**")
            else: st.success(f"✅ **{pd.to_datetime(j['datetime']).strftime('%H:%M')} | {j['title']}**")
        if not jours: st.caption("无日志规划")
        
    with c3:
        st.markdown("#### 🟠/⚪ 传代排期")
        scheds = sorted([s for s in st.session_state.schedule if pd.to_datetime(s["obs_time"]).date() == selected_d], key=lambda x: x['obs_time'])
        for s in scheds:
            if s.get("status") == "✅ 已完成":
                st.success(f"**{pd.to_datetime(s['obs_time']).strftime('%H:%M')} | {s['profile']}** (已完成)\n\n{s['details']}")
            else:
                st.error(f"**{pd.to_datetime(s['obs_time']).strftime('%H:%M')} | {s['profile']}**\n\n{s['details']}")
        if not scheds: st.caption("无传代操作")
        
    st.divider()
    st.markdown("### ✍️ 快捷随手记 (录入至当前选中日期)")
    c_m, c_j = st.columns(2)
    
    with c_m:
        with st.form("quick_memo_form", clear_on_submit=True):
            st.markdown("➕ **存入该日备忘录**")
            m_t = st.text_input("备忘录标题")
            m_time = st.time_input("时间", value=datetime.now().time())
            m_c = st.text_area("内容 (选填)")
            if st.form_submit_button("💾 快速保存备忘", use_container_width=True):
                if m_t:
                    dt_str = datetime.combine(selected_d, m_time).strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.memo.append({"id": str(uuid.uuid4()), "title": m_t, "time": dt_str, "content": m_c, "status": "pending"})
                    _save_memo()
                    st.toast("✅ 备忘录保存成功！")
                    st.rerun()

    with c_j:
        with st.form("quick_journal_form", clear_on_submit=True):
            st.markdown("➕ **新建该日实验计划**")
            j_t = st.text_input("日志/实验标题")
            j_time = st.time_input("时间", value=datetime.now().time())
            j_c = st.text_area("实验安排/步骤")
            if st.form_submit_button("💾 快速保存计划", use_container_width=True):
                if j_t:
                    dt_str = datetime.combine(selected_d, j_time).strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.journal.append({"id": str(uuid.uuid4()), "title": j_t, "datetime": dt_str, "content": j_c, "status": "⏳ 待执行", "record": ""})
                    _save_journal()
                    st.toast("✅ 实验计划保存成功！")
                    st.rerun()

# --- 新增模块：动物实验追踪 ---
def render_animal():
    st.title("🐁 动物模型与实验队列追踪")
    st.markdown("记录活体实验造模进度、干预给药节点和表型观察状态。")
    
    with st.expander("➕ 建立新动物队列", expanded=True):
        a_c1, a_c2, a_c3 = st.columns(3)
        cohort = a_c1.text_input("队列名称/分组", placeholder="如: 血管钙化小鼠模型组A")
        start_d = a_c2.date_input("开始日期/造模日")
        treat = a_c3.text_input("干预方式/特殊处理")
        if st.button("💾 保存队列"):
            if cohort:
                st.session_state.animals.append({
                    "id": str(uuid.uuid4()), "cohort": cohort,
                    "start_date": start_d.strftime("%Y-%m-%d"),
                    "treatment": treat, "status": "造模/观察中"
                })
                _save_animals()
                st.toast("✅ 新队列已创建！")
                st.rerun()
                
    st.divider()
    if st.session_state.animals:
        st.subheader("当前动物队列管理")
        df_ani = pd.DataFrame(st.session_state.animals)
        edited_ani = st.data_editor(
            df_ani, use_container_width=True, num_rows="dynamic",
            column_config={"status": st.column_config.SelectboxColumn("当前状态", options=["造模/观察中", "给药干预阶段", "已取材/封存", "异常终止"])},
            key="ani_edit"
        )
        if st.button("💾 同步动物队列数据", type="primary"):
            st.session_state.animals = edited_ani.to_dict('records')
            _save_animals()
            st.toast("✅ 动物队列状态已更新！")
    else:
        st.info("当前暂无动物队列记录。")

# --- 新增模块：生信与测序流转 ---
def render_bioinfo():
    st.title("💻 生信分析与组学数据流转")
    st.markdown("有效管理庞大的分析流水线路径、公共数据库挖掘进度及代码版本。")
    
    with st.expander("➕ 登记新分析项目", expanded=True):
        b_name = st.text_input("项目/分析流水线名称", placeholder="如: 单细胞与空间转录组联合分析")
        b_path = st.text_input("服务器代码/数据路径", placeholder="e.g., /data/user/scRNA_project_2026")
        b_stage = st.selectbox("当前分析阶段", ["测序下机/数据获取", "数据质控 (QC) / 去批次", "降维聚类 / 差异基因", "亚群注释 / 轨迹推断", "空间共定位 / 细胞通讯", "深度学习模型训练", "结果封存"])
        if st.button("💾 登记生信记录"):
            if b_name:
                st.session_state.bioinfo.append({
                    "id": str(uuid.uuid4()), "project": b_name,
                    "server_path": b_path, "stage": b_stage,
                    "update_time": datetime.now().strftime("%Y-%m-%d")
                })
                _save_bioinfo()
                st.toast("✅ 生信记录已添加！")
                st.rerun()
                
    st.divider()
    if st.session_state.bioinfo:
        st.subheader("分析流转进度板")
        df_bio = pd.DataFrame(st.session_state.bioinfo)
        edited_bio = st.data_editor(
            df_bio, use_container_width=True, num_rows="dynamic",
            column_config={"stage": st.column_config.SelectboxColumn("阶段", options=["测序下机/数据获取", "数据质控 (QC) / 去批次", "降维聚类 / 差异基因", "亚群注释 / 轨迹推断", "空间共定位 / 细胞通讯", "深度学习模型训练", "结果封存"])},
            key="bio_edit"
        )
        if st.button("💾 保存路径与分析状态更新", type="primary"):
            st.session_state.bioinfo = edited_bio.to_dict('records')
            _save_bioinfo()
            st.toast("✅ 分析进度已同步落盘！")
    else:
        st.info("当前暂无生信项目流转。")

# --- 新增模块：SOP与结果归档 ---
def render_sop_and_results():
    st.title("📚 标准方法库 (SOP) 与结果归档")
    tab1, tab2 = st.tabs(["📑 常用实验方法 (SOP) 知识库", "📂 阶段性结果与关键数据存档"])

    with tab1:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("➕ 录入新 SOP")
            s_cat = st.selectbox("分类", ["细胞实验", "动物造模", "分子生化", "生信/代码流程", "其它"])
            s_title = st.text_input("方法名称", placeholder="如: 视丘下部组织切片免疫荧光 Protocol")
            s_content = st.text_area("详细步骤 (完全支持 Markdown 语法)", height=250)
            if st.button("💾 保存至知识库", use_container_width=True):
                if s_title and s_content:
                    st.session_state.sops.append({
                        "id": str(uuid.uuid4()), "category": s_cat, "title": s_title, "content": s_content
                    })
                    _save_sops()
                    st.toast("✅ SOP入库成功！")
                    st.rerun()
        with c2:
            st.subheader("📖 知识库检索")
            if st.session_state.sops:
                sops_df = pd.DataFrame(st.session_state.sops)
                sel_cat = st.selectbox("按分类筛选", ["全部"] + list(sops_df['category'].unique()))
                filter_df = sops_df if sel_cat == "全部" else sops_df[sops_df['category'] == sel_cat]
                
                for _, row in filter_df.iterrows():
                    with st.expander(f"[{row['category']}] {row['title']}"):
                        st.markdown(row['content'])
                        if st.button("🗑️ 删除此方法", key=f"del_sop_{row['id']}"):
                            st.session_state.sops = [s for s in st.session_state.sops if s['id'] != row['id']]
                            _save_sops()
                            st.rerun()
            else:
                st.info("知识库空空如也，快去记录你的第一个 Protocol 吧！")

    with tab2:
        with st.expander("➕ 添加新实验结果归档", expanded=True):
            r_title = st.text_input("结果摘要/结论", placeholder="如: 某某基因敲除后表现出明显表型差异")
            r_loc = st.text_input("原始数据/图表存放位置", placeholder="如: 移动硬盘2/2026数据/WB图_0317")
            r_conc = st.text_area("核心结论与下一步计划探讨")
            if st.button("💾 归档入库"):
                if r_title:
                    st.session_state.results.append({
                        "id": str(uuid.uuid4()), "date": datetime.now().strftime("%Y-%m-%d"),
                        "title": r_title, "location": r_loc, "conclusion": r_conc
                    })
                    _save_results()
                    st.toast("✅ 实验结果已永久归档！")
                    st.rerun()
                    
        if st.session_state.results:
            st.subheader("🗂️ 历史归档列表")
            for r in sorted(st.session_state.results, key=lambda x: x['date'], reverse=True):
                with st.container(border=True):
                    st.markdown(f"#### 🏆 {r['title']} `({r['date']})`")
                    st.caption(f"📁 **原始数据追踪**: {r['location']}")
                    st.write(r['conclusion'])
                    if st.button("🗑️ 删除记录", key=f"del_res_{r['id']}"):
                        st.session_state.results = [i for i in st.session_state.results if i['id'] != r['id']]
                        _save_results()
                        st.rerun()

def render_inventory():
    st.title("📦 实验室库存管理系统")
    st.markdown("提示：直接在下方表格内**双击即可修改**试剂名称、余量和位置，右侧勾选复选框可删除条目，底部空行可直接新增。")
    
    df_inv = pd.DataFrame(st.session_state.inventory)
    
    # 将字典转为DF并使用数据编辑器
    edited_df = st.data_editor(
        df_inv, 
        num_rows="dynamic", 
        use_container_width=True, 
        key="inv_editor",
        column_config={
            "type": st.column_config.SelectboxColumn("分类", options=["❄️ 冻存细胞", "🧪 试剂/抗体/引物/其他", "🧃 自配培养基"]),
            "amount": st.column_config.NumberColumn("余量", min_value=0),
            "date": st.column_config.DateColumn("入库时间")
        }
    )
    
    if st.button("💾 确认并保存库存变更", type="primary"):
        # 处理空值或异常值
        edited_df = edited_df.dropna(subset=['name'])
        if 'id' not in edited_df.columns:
            edited_df['id'] = [str(uuid.uuid4()) for _ in range(len(edited_df))]
        else:
            edited_df['id'] = edited_df['id'].apply(lambda x: str(uuid.uuid4()) if pd.isna(x) else x)
            
        st.session_state.inventory = edited_df.to_dict(orient="records")
        _save_inventory()
        st.toast("✅ 库存数据已同步落盘！")

def render_journal():
    st.title("📓 实验日志与排期管理")
    interactive_calendar("journal")
    st.divider()
    
    selected_d = st.session_state.get("cal_selected_date", datetime.now().date())
    st.subheader(f"📅 {selected_d.strftime('%Y年%m月%d日')} 日志与计划")
    
    with st.expander("➕ 创建该日新实验计划", expanded=False):
        with st.form("add_journal_form", clear_on_submit=True):
            exp_title = st.text_input("实验项目名称 (例如：A549 增殖曲线)")
            exp_time = st.time_input("计划时间", value=datetime.now().time())
            exp_content = st.text_area("实验计划内容/步骤")
            if st.form_submit_button("💾 保存计划", use_container_width=True) and exp_title:
                st.session_state.journal.append({
                    "id": str(uuid.uuid4()), "title": exp_title,
                    "datetime": datetime.combine(selected_d, exp_time).strftime("%Y-%m-%d %H:%M:%S"),
                    "content": exp_content, "status": "⏳ 待执行", "record": ""
                })
                _save_journal()
                st.toast("✅ 实验计划已建立")
                st.rerun()

    tab_todo, tab_done = st.tabs(["📝 待办计划", "🗂️ 已归档日志"])
    
    with tab_todo:
        pending = [j for j in st.session_state.journal if pd.to_datetime(j["datetime"]).date() == selected_d and j["status"] == "⏳ 待执行"]
        if not pending: st.info("这天没有待执行的实验计划哟。")
        for exp in sorted(pending, key=lambda x: x["datetime"]):
            with st.container(border=True):
                st.markdown(f"#### 🧪 {exp['title']}")
                st.caption(f"📅 计划执行时间：{exp['datetime']}")
                st.info(exp['content'] if exp['content'] else "无详细步骤规划。")
                
                exp_record = st.text_area("填写实验结果/数据结论：", key=f"record_{exp['id']}")
                c1, c2 = st.columns(2)
                if c1.button("✅ 填写完毕并归档", key=f"btn_done_{exp['id']}", use_container_width=True):
                    exp["status"] = "✅ 已完成"
                    exp["record"] = exp_record if exp_record.strip() else "无额外实验记录。"
                    _save_journal()
                    st.toast("✅ 日志已归档！")
                    st.rerun()
                if c2.button("🗑️ 取消计划", key=f"btn_del_{exp['id']}", use_container_width=True):
                    st.session_state.journal = [j for j in st.session_state.journal if j["id"] != exp["id"]]
                    _save_journal()
                    st.rerun()

    with tab_done:
        done = [j for j in st.session_state.journal if pd.to_datetime(j["datetime"]).date() == selected_d and j["status"] == "✅ 已完成"]
        if not done: st.info("该日尚无已归档的日志记录信息。")
        for exp in sorted(done, key=lambda x: x["datetime"], reverse=True):
            with st.expander(f"✅ {exp['title']} ({exp['datetime']})", expanded=False):
                st.markdown("**📌 【初始计划】**\n" + (exp['content'] if exp['content'] else "空"))
                st.markdown("**🖋️ 【实验结果】**")
                st.success(exp['record'])
                c1, c2 = st.columns(2)
                if c1.button("🗑️ 彻底删除", key=f"btn_deldone_{exp['id']}", use_container_width=True):
                    st.session_state.journal = [j for j in st.session_state.journal if j["id"] != exp["id"]]
                    _save_journal()
                    st.rerun()
                if c2.button("↩️ 退回未归档 (恢复待办)", key=f"btn_undone_{exp['id']}", use_container_width=True):
                    for j in st.session_state.journal:
                        if j["id"] == exp["id"]:
                            j["status"] = "⏳ 待执行"
                            break
                    _save_journal()
                    st.rerun()

def render_memo():
    st.title("📆 个人备忘录与日历")
    interactive_calendar("memo")
    st.divider()
    
    selected_d = st.session_state.get("cal_selected_date", datetime.now().date())
    st.subheader(f"📋 {selected_d.strftime('%Y年%m月%d日')} 备忘清单")
    
    with st.expander("➕ 添加属于选中日期的备忘 / 待办", expanded=False):
        with st.form("add_memo_form", clear_on_submit=True):
            m_title = st.text_input("标题 (如: 查阅最新文献或修改基金本子)")
            m_time = st.time_input("提醒时间", value=datetime.now().time())
            m_content = st.text_area("详细内容 (选填)")
            if st.form_submit_button("💾 存入并加入日程", use_container_width=True) and m_title:
                st.session_state.memo.append({
                    "id": str(uuid.uuid4()), "title": m_title, 
                    "time": datetime.combine(selected_d, m_time).strftime("%Y-%m-%d %H:%M:%S"), "content": m_content, "status": "pending"
                })
                _save_memo()
                st.toast("✅ 备忘录已更新！")
                st.rerun()

    tab_todo_m, tab_done_m = st.tabs(["📝 待办备忘", "🗂️ 已完成归档"])
    
    with tab_todo_m:
        active_memos = sorted([m for m in st.session_state.memo if pd.to_datetime(m["time"]).date() == selected_d and m.get("status", "pending") != "completed"], key=lambda x: x['time'])
        if not active_memos: st.info("所选日期下无待办备忘记录！")
        for m in active_memos:
            with st.container(border=True):
                c1, c2, c3 = st.columns([4, 1, 1])
                with c1:
                    st.markdown(f"**⏰ {pd.to_datetime(m['time']).strftime('%H:%M')} | {m['title']}**")
                    st.caption(m['content'] if m['content'] else "_无详细内容_")
                with c2:
                    if st.button("✅ 完成", key=f"done_memo_{m['id']}", use_container_width=True):
                        m["status"] = "completed"
                        _save_memo()
                        st.rerun()
                with c3:
                    if st.button("🗑️ 删除", key=f"del_memo_{m['id']}", use_container_width=True):
                        st.session_state.memo = [im for im in st.session_state.memo if im['id'] != m['id']]
                        _save_memo()
                        st.rerun()

    with tab_done_m:
        done_memos = sorted([m for m in st.session_state.memo if pd.to_datetime(m["time"]).date() == selected_d and m.get("status", "pending") == "completed"], key=lambda x: x['time'], reverse=True)
        if not done_memos: st.info("所选日期下无已归档备忘记录！")
        for m in done_memos:
            with st.container(border=True):
                c1, c2, c3 = st.columns([4, 1, 1])
                with c1:
                    st.markdown(f"**⏰ {pd.to_datetime(m['time']).strftime('%H:%M')} | [已完成] {m['title']}**")
                    st.caption(m['content'] if m['content'] else "_无详细内容_")
                with c2:
                    if st.button("↩️ 退回待办", key=f"undone_memo_{m['id']}", use_container_width=True):
                        m["status"] = "pending"
                        _save_memo()
                        st.rerun()
                with c3:
                    if st.button("🗑️ 删除", key=f"deldone_memo_{m['id']}", use_container_width=True):
                        st.session_state.memo = [im for im in st.session_state.memo if im['id'] != m['id']]
                        _save_memo()
                        st.rerun()

def render_cell_kinetics():
    st.sidebar.title("🧬 细胞档案管理")
    with st.sidebar.form("add_profile_form", clear_on_submit=True):
        st.subheader("➕ 添加新细胞/血清配置")
        new_cell_name = st.text_input("细胞系名称 (如: VSMC)")
        new_fbs = st.text_input("FBS 浓度 (如: 5%)")
        if st.form_submit_button("创建新档案"):
            if new_cell_name and new_fbs:
                profile_name = f"{new_cell_name}_{new_fbs}FBS"
                if profile_name not in st.session_state.cell_db:
                    st.session_state.cell_db[profile_name] = {"data": pd.DataFrame(columns=["N0", "t", "Nt", "timestamp"]), "r": 0.05}
                    save_cell_db()
                    st.toast(f"✅ 已成功创建档案: {profile_name}")
                else:
                    st.error("该档案已存在！")

    profile_list = list(st.session_state.cell_db.keys())
    if not profile_list:
        st.info("请先在左侧侧边栏创建一个细胞档案。")
        return

    current_profile = st.sidebar.selectbox("👉 选择当前使用的细胞档案", profile_list)
    profile_data = st.session_state.cell_db[current_profile]

    st.title("🧫 专属细胞传代动力学预测系统")
    tab1, tab2, tab3, tab4 = st.tabs(["🧮 传代计算器", "📈 数据录入校准", "📅 传代日程表", "🗂️ 传代归档汇总"])

    if 'calc_history' not in st.session_state: st.session_state.calc_history = []

    # --- TAB 1: 传代计算器 ---
    with tab1:
        st.markdown(f"**当前模型：`{current_profile}`** (专属生长常数 $r$ ≈ {profile_data['r']:.4f})")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("传前状态 (源容器)")
            src_vessel = st.selectbox("源容器类型", list(AREA_MAP.keys()), index=0)
            src_density = st.number_input("当前细胞密度 (%)", min_value=10, max_value=120, value=90, step=5)
        with c2:
            st.subheader("目标期望 (目标容器)")
            tgt_vessel = st.selectbox("目标容器类型", list(AREA_MAP.keys()), index=4)
            tgt_time = st.number_input("生长时间 (小时)", min_value=6, max_value=120, value=24, step=6)
            obs_dt = datetime.now() + timedelta(hours=tgt_time)
            st.info(f"👉 **预计观察时间**: {obs_dt.strftime('%Y年%m月%d日 %H:%M')}")
            tgt_density = st.number_input("期望达到的密度 (%)", min_value=30, max_value=100, value=85, step=5)

        if st.button("🚀 开始计算传代方案", use_container_width=True):
            r_val = profile_data['r']
            required_N0 = calculate_inverse_N0(tgt_density, tgt_time, r_val)
            passage_ratio = (AREA_MAP[tgt_vessel] * required_N0) / (AREA_MAP[src_vessel] * src_density)
            plan = {
                "profile": current_profile, "src_vessel": src_vessel, "src_density": src_density, 
                "tgt_vessel": tgt_vessel, "tgt_time": tgt_time, "tgt_density": tgt_density, 
                "required_N0": required_N0, "passage_ratio": passage_ratio,
                "calc_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.proposed_plan = plan
            st.session_state.calc_history.append(plan)

        if "proposed_plan" in st.session_state and st.session_state.proposed_plan.get("profile") == current_profile:
            plan = st.session_state.proposed_plan
            st.divider()
            if plan["passage_ratio"] > 1.0:
                st.error(f"❌ 细胞数量不足！计算所需比例为 {plan['passage_ratio']*100:.1f}%，超过了 100%。")
            else:
                st.success(f"✅ 目标容器需要的接种密度为：**{plan['required_N0']:.1f}%**")
                match = re.search(r"(\d+)孔板 \(单孔\)", plan["tgt_vessel"])
                if match:
                    whole_ratio = plan["passage_ratio"] * int(match.group(1))
                    extra_msg = f"\n\n*(如需铺满整板，需消耗源容器 {whole_ratio*100:.1f}% 的细胞)*"
                else:
                    extra_msg = ""
                
                st.info(f"👉 **传代吸取比例：{plan['passage_ratio']*100:.1f}%**" + extra_msg)
                
                if st.button("✅ 确认方案并加入排期日程", use_container_width=True):
                    now = datetime.now()
                    obs_time = now + timedelta(hours=plan['tgt_time'])
                    st.session_state.schedule.append({
                        "id": str(uuid.uuid4()), "profile": current_profile, 
                        "start_time": now.strftime("%Y-%m-%d %H:%M:%S"), "obs_time": obs_time.strftime("%Y-%m-%d %H:%M:%S"), 
                        "details": f"从 {plan['src_vessel']} 传代培养至 {plan['tgt_vessel']}。目标密度：{plan['tgt_density']}%。计算传代比例: {plan['passage_ratio']*100:.1f}%。"
                    })
                    _save_schedule()
                    del st.session_state.proposed_plan
                    st.toast("🎉 已成功存入排期日程表！")
                    st.rerun()
                    
        if st.session_state.calc_history:
            with st.expander("🕒 传代方案计算历史"):
                hist_data = [{"计算时间": h["calc_time"], "细胞系": h["profile"], "源容器": h["src_vessel"], "目标容器": h["tgt_vessel"], "目标密度": f"{h['tgt_density']}%", "生长时间": f"{h['tgt_time']}h", "推荐比例": f"{h['passage_ratio']*100:.1f}%"} for h in reversed(st.session_state.calc_history)]
                st.dataframe(pd.DataFrame(hist_data), use_container_width=True)

    # --- TAB 2: 录入与拟合 ---
    with tab2:
        st.markdown("填入真实的传代结果，让背后的数学模型不断进行 $r$ 值迭代优化。")
        df = profile_data['data']
        if not df.empty:
            edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, key=f"data_editor_{current_profile}")
            if st.button("💾 应用表格更改", key="btn_apply_df"):
                st.session_state.cell_db[current_profile]['data'] = edited_df.dropna(how='all').reset_index(drop=True)
                save_cell_db()
                st.rerun()
        
        st.subheader("📝 录入新观测数据")
        with st.form("add_data_form", clear_on_submit=False):
            c1, c2, c3 = st.columns(3)
            with c1: rec_src = st.selectbox("传代前容器", list(AREA_MAP.keys()), index=0)
            with c2: rec_den = st.number_input("传前密度 (%)", value=90.0)
            with c3: rec_rat = st.number_input("实际传代比例 (%)", value=33.3)
            
            c4, c5, c6 = st.columns(3)
            with c4: rec_tgt = st.selectbox("传代后容器", list(AREA_MAP.keys()), index=4)
            with c5: rec_t = st.number_input("生长时间 (小时)", value=24.0)
            with c6: rec_Nt = st.number_input("最终长成密度 (%)", value=80.0)
            
            if st.form_submit_button("💾 保存并校准模型"):
                calc_N0 = (rec_den * AREA_MAP[rec_src] * (rec_rat / 100.0)) / AREA_MAP[rec_tgt]
                new_row = pd.DataFrame({"N0": [calc_N0], "t": [rec_t], "Nt": [rec_Nt], "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]})
                updated_df = pd.concat([st.session_state.cell_db[current_profile]['data'], new_row], ignore_index=True)
                st.session_state.cell_db[current_profile]['data'] = updated_df
                
                if len(updated_df) >= 1:
                    try:
                        X_data = (updated_df['N0'].values, updated_df['t'].values)
                        y_data = updated_df['Nt'].values
                        weights = np.linspace(1.0, 5.0, len(updated_df))
                        popt, _ = curve_fit(logistic_model, X_data, y_data, p0=[max(0.001, st.session_state.cell_db[current_profile]['r'])], bounds=(0.001, 0.5), sigma=1.0/np.sqrt(weights))
                        st.session_state.cell_db[current_profile]['r'] = popt[0]
                        st.toast(f"模型优化成功！最新 $r$ = {popt[0]:.4f}")
                    except Exception:
                        st.error("拟合公式参数失败，可能是数据点分布异常。数据已保存，请检查。")
                save_cell_db()
                st.rerun()

        st.divider()
        st.subheader("📝 传代观察前信息记录表")
        st.markdown("在此自由记录传代操作信息，作为观察前的辅助参考。")
        if 'pre_passage_records' not in st.session_state.cell_db[current_profile]:
            st.session_state.cell_db[current_profile]['pre_passage_records'] = pd.DataFrame(columns=["传代前容器", "传前密度 (%)", "传代比例 (%)", "传代后容器", "日期"])
        
        pre_rec_df = st.session_state.cell_db[current_profile]['pre_passage_records']
        edited_pre_rec = st.data_editor(pre_rec_df, num_rows="dynamic", use_container_width=True, key=f"pre_passage_editor_{current_profile}")
        if st.button("💾 保存记录表格", key="btn_save_pre_rec"):
            st.session_state.cell_db[current_profile]['pre_passage_records'] = edited_pre_rec.dropna(how='all')
            save_cell_db()
            st.toast("✅ 记录已保存！")

    # --- TAB 3: 传代日程 ---
    with tab3:
        st.subheader("🗓️ 细胞实验及传代排期 (🟠 待办 | ⚪ 已完成)")
        interactive_calendar("schedule")
        st.divider()
        
        selected_d = st.session_state.get("cal_selected_date", datetime.now().date())
        scheds = sorted([s for s in st.session_state.schedule if pd.to_datetime(s["obs_time"]).date() == selected_d], key=lambda x: x['obs_time'])
        
        if not scheds: st.info("该天系统尚未排期。")
        for s in scheds:
            is_done = s.get("status") == "✅ 已完成"
            with st.container(border=True):
                if is_done:
                    st.markdown(f"**⏰ {pd.to_datetime(s['obs_time']).strftime('%H:%M')} | [已完成] {s['profile']}**")
                    st.write(s['details'])
                    st.success(s.get('record', '无额外日志。'))
                    if st.button("↩️ 退回未归档", key=f"unarch_sched_{s['id']}", use_container_width=True):
                        s['status'] = "⏳ 待执行"
                        _save_schedule()
                        st.rerun()
                else:
                    st.markdown(f"**⏰ {pd.to_datetime(s['obs_time']).strftime('%H:%M')} | [档案] {s['profile']}**")
                    st.write(s['details'])
                    log_text = st.text_area("实验日志记录", placeholder="在此记录操作情况...", key=f"log_text_s_{s['id']}")
                    
                    c1, c2, c3 = st.columns([3, 3, 4])
                    if c1.button("✅ 完成并归档", key=f"done_s_{s['id']}", use_container_width=True):
                        s["status"] = "✅ 已完成"
                        s["record"] = log_text if log_text.strip() else "无额外日志。"
                        s["finish_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        _save_schedule()
                        st.toast("✅ 排期记录已归档")
                        st.rerun()
                    if c2.button("🗑️ 删除排期", key=f"del_s_{s['id']}", use_container_width=True):
                        st.session_state.schedule = [si for si in st.session_state.schedule if si['id'] != s['id']]
                        _save_schedule()
                        st.rerun()
                    if "batch_id" in s and c3.button("🗑️ 批量删同批次", key=f"del_batch_s_{s['id']}", use_container_width=True):
                        st.session_state.schedule = [si for si in st.session_state.schedule if si.get('batch_id') != s['batch_id']]
                        _save_schedule()
                        st.rerun()

        st.divider()
        with st.expander("➕ 批量生成：多次用药实验日历规划", expanded=False):
            with st.form("batch_schedule_form", clear_on_submit=True):
                plan_name = st.text_input("实验任务备注", value="药物浓度梯度验证用药")
                c1, c2 = st.columns(2)
                with c1:
                    plating_time = st.time_input("首次铺板执行时间表", value=datetime.now().time())
                    drug_days = st.number_input("铺板后几天开始加药", value=1)
                with c2:
                    drug_interval = st.number_input("加药间隔频率 (天)", value=2.0)
                    harvest_days = st.number_input("总共几天后完结收样", value=5)
                    
                if st.form_submit_button("✅ 添加生成长线日程", use_container_width=True):
                    base_dt = datetime.combine(selected_d, plating_time)
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    batch_id = str(uuid.uuid4())
                    
                    st.session_state.schedule.append({"id": str(uuid.uuid4()), "batch_id": batch_id, "batch_name": plan_name, "profile": current_profile, "start_time": now_str, "obs_time": base_dt.strftime("%Y-%m-%d %H:%M:%S"), "details": f"[{plan_name}] 🧪 开展基础铺板。"})
                    drug_dt = base_dt + timedelta(days=drug_days)
                    h_dt = base_dt + timedelta(days=harvest_days)
                    
                    if drug_interval > 0:
                        curr_d = drug_dt
                        drug_count = 1
                        while curr_d < h_dt:
                            st.session_state.schedule.append({"id": str(uuid.uuid4()), "batch_id": batch_id, "batch_name": plan_name, "profile": current_profile, "start_time": now_str, "obs_time": curr_d.strftime("%Y-%m-%d %H:%M:%S"), "details": f"[{plan_name}] 💉 第 {drug_count} 次换药/加药"})
                            curr_d += timedelta(days=drug_interval)
                            drug_count += 1
                    else:
                        if drug_dt < h_dt:
                            st.session_state.schedule.append({"id": str(uuid.uuid4()), "batch_id": batch_id, "batch_name": plan_name, "profile": current_profile, "start_time": now_str, "obs_time": drug_dt.strftime("%Y-%m-%d %H:%M:%S"), "details": f"[{plan_name}] 💉 单次加药"})
                    
                    st.session_state.schedule.append({"id": str(uuid.uuid4()), "batch_id": batch_id, "batch_name": plan_name, "profile": current_profile, "start_time": now_str, "obs_time": h_dt.strftime("%Y-%m-%d %H:%M:%S"), "details": f"[{plan_name}] 🧬 检测收样完结。"})
                    _save_schedule()
                    st.toast("✅ 批量长线规划安排完毕！")
                    st.rerun()

    # --- TAB 4: 归档汇总 ---
    with tab4:
        st.subheader(f"🗂️ {current_profile} 专属操作归档")
        st.markdown("这里汇总了该细胞系所有已经标记为「✅ 完成」的传代记录。")

        target_title_prefix = f"【排期归档】{current_profile}"
        archived_journal_logs = [j for j in st.session_state.journal if target_title_prefix in j['title'] and j['status'] == '✅ 已完成']
        archived_schedules = [s for s in st.session_state.schedule if s.get('profile') == current_profile and s.get('status') == '✅ 已完成']
        
        all_archived = []
        for j in archived_journal_logs:
            all_archived.append({"type": "journal", "data": j, "sort_time": j["datetime"]})
        for s in archived_schedules:
            all_archived.append({"type": "schedule", "data": s, "sort_time": s.get("finish_time", s["obs_time"])})
        
        if not all_archived:
            st.info("尚未有适用于该细胞系的已归档历史。")
        else:
            for item in sorted(all_archived, key=lambda x: x["sort_time"], reverse=True):
                if item["type"] == "journal":
                    log = item["data"]
                    with st.expander(f"✅ {log['datetime']} | 操作记录 (旧版归档)", expanded=False):
                        st.markdown("**📌 【操作事项】**\n" + (log['content'] if log['content'] else "无操作详情"))
                        st.markdown("**🖋️ 【实验结果记录】**")
                        st.success(log['record'])
                        c1, c2 = st.columns(2)
                        if c1.button("🗑️ 删除此归档", key=f"del_arch_{log['id']}", use_container_width=True):
                            st.session_state.journal = [j for j in st.session_state.journal if j["id"] != log["id"]]
                            _save_journal()
                            st.rerun()
                        if c2.button("↩️ 退回未归档 (迁移至日程表)", key=f"unarch_{log['id']}", use_container_width=True):
                            st.session_state.schedule.append({
                                "id": log["id"], "profile": current_profile,
                                "start_time": log["datetime"], "obs_time": log["datetime"], 
                                "details": log["content"], "status": "⏳ 待执行"
                            })
                            st.session_state.journal = [j for j in st.session_state.journal if j["id"] != log["id"]]
                            _save_journal()
                            _save_schedule()
                            st.rerun()
                else:
                    s = item["data"]
                    disp_time = s.get("finish_time", s["obs_time"])
                    with st.expander(f"✅ {disp_time} | 操作记录", expanded=False):
                        st.markdown("**📌 【操作事项】**\n" + (s['details'] if s['details'] else "无操作详情"))
                        st.markdown("**🖋️ 【实验结果记录】**")
                        st.success(s.get('record', '无额外日志。'))
                        c1, c2 = st.columns(2)
                        if c1.button("🗑️ 删除此归档", key=f"del_arch_s_{s['id']}", use_container_width=True):
                            st.session_state.schedule = [si for si in st.session_state.schedule if si["id"] != s["id"]]
                            _save_schedule()
                            st.rerun()
                        if c2.button("↩️ 退回未归档 (待办)", key=f"unarch_s_{s['id']}", use_container_width=True):
                            for si in st.session_state.schedule:
                                if si["id"] == s["id"]:
                                    si["status"] = "⏳ 待执行"
                                    break
                            _save_schedule()
                            st.rerun()

# ==========================================
# 5. 主程序入口与导航路由
# ==========================================
def main():
    st.sidebar.title("🧭 科研管理导航")
    
    # 构建侧边栏路由字典，简化判断逻辑
    pages = {
        "🏠 首页大屏 (待办提醒)": render_home,
        "🧫 细胞传代动力学系统": render_cell_kinetics,
        "🐁 动物模型与实验追踪": render_animal,
        "💻 生信分析与测序流转": render_bioinfo,
        "📚 SOP与实验结果归档": render_sop_and_results,
        "📦 实验室试剂库存管理": render_inventory,
        "📓 实验日志与排期": render_journal,
        "📆 个人备忘日历": render_memo
    }
    
    app_mode = st.sidebar.radio("选择进入模块", list(pages.keys()))
    st.sidebar.divider()
    
    # 动态渲染被选中的页面
    pages[app_mode]()

if __name__ == "__main__":
    main()

