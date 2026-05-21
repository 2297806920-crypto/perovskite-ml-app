import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, LeaveOneOut
import matplotlib.pyplot as plt
import re
from scipy.interpolate import make_interp_spline

st.set_page_config(page_title="钙钛矿催化智能设计", page_icon="🔬", layout="wide")
st.title("🔬 钙钛矿光热催化智能设计平台")
st.markdown("### 上传预计算的特征文件，一键训练与预测")
st.markdown("---")

with st.sidebar:
    st.header("📂 上传特征文件")
    uploaded_file = st.file_uploader("上传 features.csv（需包含 formula 和 target 列）", type=["csv"])
    if uploaded_file:
        st.success("文件已上传！")
    st.markdown("---")
    st.markdown("### ⚙️ 如何准备特征文件？")
    st.markdown("在本地用 `generate_features.py` 处理原始数据，生成包含132维Magpie特征和`target`列的CSV。")

if uploaded_file is None:
    st.info("👈 请从左侧上传特征文件开始。")
else:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 数据预览")
    st.dataframe(df.head(10))
    
    # 检查必要列
    if 'target' not in df.columns:
        st.error("缺少 `target` 列！请确保特征文件中包含目标值。")
        st.stop()
    if 'formula' not in df.columns:
        st.error("缺少 `formula` 列！")
        st.stop()

    # 准备特征和目标
    feature_cols = [c for c in df.columns if c not in ['formula', 'target']]
    X = df[feature_cols].values
    y = df['target'].values

    st.subheader("🧹 数据清洗")
    mask = ~np.isnan(y)
    X, y = X[mask], y[mask]
    st.write(f"有效数据：{len(y)} 条，特征维度：{X.shape[1]}")

    if len(y) == 0:
        st.error("无有效目标值，无法训练。")
        st.stop()

    # 训练模型
    st.subheader("🌲 模型训练")
    with st.spinner("训练随机森林..."):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        if len(y) >= 3:
            loo = LeaveOneOut()
            scores = cross_val_score(model, X, y, cv=loo, scoring='r2')
            st.metric("留一法 R²", f"{scores.mean():.3f} (±{scores.std():.3f})")
        else:
            st.warning(f"数据量仅 {len(y)} 条，跳过交叉验证")
        model.fit(X, y)

    # 特征重要性
    st.subheader("📊 特征重要性")
    importances = model.feature_importances_
    top_n = 10
    top_idx = np.argsort(importances)[-top_n:][::-1]
    top_features = [feature_cols[i] for i in top_idx]
    top_importances = importances[top_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(top_n), top_importances[::-1], color='steelblue')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Features")
    st.pyplot(fig)

    # 预测新组分
    st.subheader("🔮 预测新组分")
    st.markdown("输入化学式，每行一个（需已在本地生成对应特征）")
    new_formulas = st.text_area("化学式", placeholder="La0.8Ca0.2MnO3\nLa0.5Sr0.5MnO3", height=150)
    
    if st.button("🚀 预测"):
        if new_formulas.strip():
            formulas = [f.strip() for f in new_formulas.split("\n") if f.strip()]
            # 注意：这里假设已有一个包含新组分特征的CSV，实际使用时需提前准备
            st.warning("轻量版需要预先将新组分的特征生成为CSV并上传。完整版支持即时特征生成。")
            # 为演示，可上传另一个特征文件
        else:
            st.warning("请输入至少一个化学式。")

st.markdown("---")
st.caption("© 2025 钙钛矿催化智能设计平台 | 轻量部署版")
