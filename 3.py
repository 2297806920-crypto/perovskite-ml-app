import streamlit as st
import pandas as pd
import numpy as np
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, LeaveOneOut
import matplotlib.pyplot as plt

# ==================== 页面设置 ====================
st.cache_data.clear()
st.set_page_config(page_title="钙钛矿光热催化智能设计平台", page_icon="🔬", layout="wide")

# ==================== 标题 ====================
st.title("🔬 钙钛矿光热催化智能设计平台")
st.markdown("### 从文献数据到新组分预测，一站式机器学习工具")
st.markdown("---")

# ==================== 侧边栏：上传数据 ====================
with st.sidebar:
    st.header("📂 数据导入")
    uploaded_file = st.file_uploader("上传你的 CSV 数据文件", type=["csv"])
    if uploaded_file:
        st.success("数据上传成功！")
    st.markdown("---")
    st.markdown("### 📋 示例数据格式")
    st.code("formula,CH4_rate,H2_rate,CO_rate,temperature,light\nLaMnO3,115,134,192,600,photothermal", language="text")

# ==================== 主区域 ====================
if uploaded_file is None:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🧪 这个平台能做什么？")
        st.markdown("""
        - 📄 自动处理文献中提取的催化性能数据
        - 🧬 生成 132 维元素描述符（Magpie 特征）
        - 🌲 训练随机森林模型，留一法交叉验证
        - 📊 展示特征重要性，揭示构效关系
        - 🔮 预测新组分的催化性能
        """)
    with col2:
        st.markdown("### 📤 开始使用")
        st.markdown("请从左侧上传你的 CSV 数据文件。")
else:
    # ==================== 读取数据 ====================
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"读取 CSV 失败：{e}")
        st.stop()

    st.subheader("📄 数据预览")
    st.dataframe(df.head(10))
    st.caption(f"共 {len(df)} 行，列名：{list(df.columns)}")

    if 'formula' not in df.columns:
        st.error("CSV 必须包含 `formula` 列！")
        st.stop()

    # ==================== 选择目标列 ====================
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("没有找到数值类型的列用于预测！")
        st.stop()

    target_col = st.selectbox("🎯 选择你要预测的目标列：", numeric_cols)

    # ==================== 数据清洗 ====================
    st.subheader("🧹 数据清洗")
    y_numeric = pd.to_numeric(df[target_col], errors='coerce')
    df_clean = df.copy()
    df_clean[target_col] = y_numeric
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=[target_col])
    after = len(df_clean)
    st.write(f"原始数据：{before} 行 → 清洗后：{after} 行（丢弃 {before - after} 行空值）")

    if after == 0:
        st.error("所有行的目标列都是空值，无法训练。")
        st.stop()

    # ==================== 生成特征 ====================
    st.subheader("🧬 特征生成")
    with st.spinner("正在生成 132 维 Magpie 特征..."):
        featurizer = ElementProperty.from_preset('magpie')
        valid_indices = []
        compositions = []
        for i, formula in enumerate(df_clean['formula']):
            try:
                comp = Composition(str(formula))
                compositions.append(comp)
                valid_indices.append(i)
            except Exception:
                st.warning(f"跳过无法解析的化学式：{formula}")

        df_valid = df_clean.iloc[valid_indices].reset_index(drop=True)
        X_list = featurizer.featurize_many(compositions)
        X = np.array(X_list)
        y = df_valid[target_col].values.astype(float)

    st.success(f"训练数据：{len(y)} 条，特征维度：{X.shape[1]}")

    # ==================== 训练模型 ====================
    st.subheader("🌲 模型训练")
    with st.spinner("正在训练随机森林模型..."):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        if len(y) >= 3:
            loo = LeaveOneOut()
            scores = cross_val_score(model, X, y, cv=loo, scoring='r2')
            st.metric("留一法交叉验证 R²", f"{scores.mean():.3f} (±{scores.std():.3f})")
        else:
            st.warning(f"数据量仅 {len(y)} 条，不足以做留一法交叉验证（至少需要 3 条）")
        model.fit(X, y)

    # ==================== 特征重要性 ====================
    st.subheader("📊 特征重要性")
    feature_names = featurizer.feature_labels()
    importances = model.feature_importances_
    top_n = 10
    top_idx = np.argsort(importances)[-top_n:][::-1]
    top_features = [feature_names[i] for i in top_idx]
    top_importances = importances[top_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(top_n), top_importances[::-1], color='steelblue')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top 10 Most Important Features")
    st.pyplot(fig)

    importance_df = pd.DataFrame({'Feature': top_features, 'Importance': top_importances})
    st.dataframe(importance_df)
        # ==================== 预测新组分 + 火山图 ====================
    st.subheader("🔮 预测新组分")
    st.markdown("输入一个或多个化学式（每行一个，或逗号分隔）：")
    new_formulas_input = st.text_area(
        "化学式",
        placeholder="La0.9Ca0.1MnO3\nLa0.8Ca0.2MnO3\nLa0.7Ca0.3MnO3\nLa0.5Ca0.5MnO3",
        height=200
    )

    if st.button("🚀 开始预测"):
        if new_formulas_input.strip():
            formulas = [f.strip() for f in new_formulas_input.replace("\n", ",").split(",") if f.strip()]
            results = []
            for f in formulas:
                try:
                    comp = Composition(f)
                    X_new = np.array(featurizer.featurize(comp)).reshape(1, -1)
                    pred = model.predict(X_new)[0]
                    results.append({"Formula": f, "Predicted": pred})
                except Exception as e:
                    st.warning(f"预测失败：{f} → {e}")

            if results:
                pred_df = pd.DataFrame(results)
                pred_df = pred_df.sort_values("Predicted", ascending=False)
                
                st.markdown("### 📊 预测结果")
                st.dataframe(pred_df)
                
                csv = pred_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 下载预测结果 CSV",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv',
                )

                # ==================== 火山图 ====================
                import re
                from scipy.interpolate import make_interp_spline

                doped_results = []
                for _, row in pred_df.iterrows():
                    formula = row['Formula']
                    pred = row['Predicted']
                    match = re.search(r'La(\d+\.?\d*)(Ca|Sr|Ba)(\d+\.?\d*)MnO3', formula)
                    if match:
                        la_content = float(match.group(1))
                        dopant = match.group(2)
                        dopant_content = float(match.group(3))
                        x = 1.0 - la_content
                        if abs(x - dopant_content) < 0.05:
                            doped_results.append({
                                'formula': formula,
                                'dopant': dopant,
                                'x': x,
                                'predicted': pred
                            })

                if doped_results:
                    st.markdown("### 📈 火山曲线")
                    doped_df = pd.DataFrame(doped_results)
                    
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    colors = {'Ca': '#3498db', 'Sr': '#e74c3c', 'Ba': '#2ecc71'}
                    
                    for dopant in doped_df['dopant'].unique():
                        subset = doped_df[doped_df['dopant'] == dopant].sort_values('x')
                        x_vals = subset['x'].values
                        y_vals = subset['predicted'].values
                        
                        if len(x_vals) >= 3:
                            x_smooth = np.linspace(x_vals.min(), x_vals.max(), 300)
                            spl = make_interp_spline(x_vals, y_vals, k=2)
                            y_smooth = spl(x_smooth)
                            ax2.plot(x_smooth, y_smooth, color=colors.get(dopant, 'gray'),
                                     linewidth=2.5, label=f'La₁₋ₓ{dopant}ₓMnO₃')
                            ax2.scatter(x_vals, y_vals, s=40, c=colors.get(dopant, 'gray'), zorder=5)
                        else:
                            ax2.plot(x_vals, y_vals, marker='o', linestyle='-', linewidth=2.5,
                                     markersize=8, color=colors.get(dopant, 'gray'),
                                     label=f'La₁₋ₓ{dopant}ₓMnO₃')
                        
                        best = subset.loc[subset['predicted'].idxmax()]
                        ax2.annotate(f"x={best['x']:.2f}\n{best['predicted']:.3f}", 
                                     xy=(best['x'], best['predicted']),
                                     xytext=(best['x']+0.05, best['predicted']+0.01),
                                     fontsize=10, color=colors.get(dopant, 'gray'),
                                     arrowprops=dict(arrowstyle='->', color=colors.get(dopant, 'gray')))

                    ax2.set_xlabel('A-site Doping Level (x)', fontsize=12)
                    ax2.set_ylabel('Predicted Performance', fontsize=12)
                    ax2.set_title('Volcano Plot: Predicted Performance vs Doping Level', fontsize=14)
                    ax2.legend(fontsize=11)
                    ax2.grid(True, alpha=0.3)
                    st.pyplot(fig2)
                    st.caption("火山曲线：横轴为A位掺杂比例x，纵轴为模型预测值。标记点为各系列最优掺杂比例。")
                else:
                    st.info("💡 提示：预测结果中未检测到 La₁₋ₓAₓMnO₃ 型掺杂系列，无法自动生成火山图。")
        else:
            st.warning("请输入至少一个化学式。")
st.markdown("---")
st.caption("© 2025 钙钛矿光热催化智能设计平台 | 指导教师：王志强 教授")
