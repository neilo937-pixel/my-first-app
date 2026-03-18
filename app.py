import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as obj
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
import io

# -- Page Config --
st.set_page_config(page_title="R&D Data Analytics Dashboard", layout="wide")

st.title("🔬 R&D 데이터 분석 및 시각화 대시보드")
st.markdown("---")

# -- Sidebar: Data Loading & Options --
st.sidebar.header("📂 데이터 로드 설정")
uploaded_file = st.sidebar.file_uploader("파일을 업로드하세요 (CSV, Excel, TXT)", type=["csv", "xlsx", "txt"])

if uploaded_file is not None:
    # Delimiter and Skip Rows Options
    file_extension = uploaded_file.name.split('.')[-1]
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        delimiter = st.selectbox("구분자 (Delimiter)", [",", "\t", ";", "Space"], index=0)
        if delimiter == "Space": delimiter = " "
    with col2:
        skip_rows = st.number_input("건너뛸 행 (Skip Rows)", min_value=0, value=0)

    try:
        # Load Data
        if file_extension == 'xlsx':
            df = pd.read_excel(uploaded_file, skiprows=skip_rows)
        else:
            df = pd.read_csv(uploaded_file, sep=delimiter, skiprows=skip_rows)

        st.success(f"✅ '{uploaded_file.name}' 로드 완료!")

        # -- Sidebar: Axis Selection --
        st.sidebar.header("🎯 축 및 변수 선택")
        columns = df.columns.tolist()
        
        x_axis = st.sidebar.selectbox("X축(시간/기준) 선택", columns, index=0)
        y_axis = st.sidebar.selectbox("Y축(분석 대상) 선택", columns, index=1 if len(columns) > 1 else 0)

        # -- Sidebar: Smoothing Options --
        st.sidebar.header("🧹 노이즈 제거 (Smoothing)")
        apply_smoothing = st.sidebar.checkbox("스무딩 필터 적용", value=True)
        window_size = st.sidebar.slider("Window Size (홀수)", 3, 51, 11, step=2)
        poly_order = st.sidebar.slider("Polynomial Order", 1, 5, 2)

        # -- Data Processing --
        # Sort by X-axis to ensure correct derivative calculation
        df = df.sort_values(by=x_axis).reset_index(drop=True)
        
        # Original Y
        y_raw = df[y_axis].values
        x_vals = df[x_axis].values

        # Smoothing
        if apply_smoothing:
            # Ensure window size is not larger than data length
            actual_window = min(window_size, len(df) - (1 if len(df) % 2 == 0 else 0))
            if actual_window < 3: actual_window = 3
            y_proc = savgol_filter(y_raw, actual_window, poly_order)
            df['Y_Processed'] = y_proc
        else:
            y_proc = y_raw
            df['Y_Processed'] = y_raw

        # -- Math: Derivatives (dx aware) --
        # np.gradient handles non-uniform spacing if x_vals is provided
        df['1st_Derivative'] = np.gradient(y_proc, x_vals)
        df['2nd_Derivative'] = np.gradient(df['1st_Derivative'], x_vals)

        # -- Layout: Main Dashboard --
        tab1, tab2, tab3 = st.tabs(["📈 시각화 분석", "📋 데이터 요약", "🔗 상관관계"])

        with tab1:
            # 1. Main Signal Plot (Original vs Smoothed)
            st.subheader(f"📊 {y_axis} 신호 분석")
            fig1 = obj.Figure()
            fig1.add_trace(obj.Scatter(x=df[x_axis], y=y_raw, name="Original", line=dict(color='rgba(100,100,100,0.3)')))
            fig1.add_trace(obj.Scatter(x=df[x_axis], y=df['Y_Processed'], name="Smoothed", line=dict(color='#1f77b4', width=2)))
            fig1.update_layout(xaxis_title=x_axis, yaxis_title=y_axis, hovermode="x unified", height=500)
            st.plotly_chart(fig1, use_container_width=True)

            # 2. Derivative Plot (Dual Y-axis)
            st.subheader("⚡ 1차/2차 미분 (변화율 및 가속도)")
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig2.add_trace(
                obj.Scatter(x=df[x_axis], y=df['Y_Processed'], name=f"{y_axis} (Ref)", line=dict(color='gray', dash='dash')),
                secondary_y=False
            )
            fig2.add_trace(
                obj.Scatter(x=df[x_axis], y=df['1st_Derivative'], name="1st Dev (Velocity)", line=dict(color='#ff7f0e', width=2)),
                secondary_y=True
            )
            fig2.add_trace(
                obj.Scatter(x=df[x_axis], y=df['2nd_Derivative'], name="2nd Dev (Accel)", line=dict(color='#2ca02c', width=1)),
                secondary_y=True
            )

            fig2.update_yaxes(title_text=f"Original {y_axis}", secondary_y=False)
            fig2.update_yaxes(title_text="Derivative Value", secondary_y=True)
            fig2.update_xaxes(title_text=x_axis)
            fig2.update_layout(hovermode="x unified", height=500)
            st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            st.subheader("📝 데이터 요약 및 이상치 탐지")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**기초 통계량:**")
                st.dataframe(df[[x_axis, y_axis, '1st_Derivative']].describe())
            with col_b:
                st.write("**결측치 확인:**")
                st.write(df.isnull().sum())
            
            st.write("**데이터 미리보기 (상위 10행):**")
            st.dataframe(df.head(10))

            # Export Button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 분석 데이터 다운로드 (CSV)",
                data=csv,
                file_name=f"processed_{uploaded_file.name}",
                mime='text/csv',
            )

        with tab3:
            st.subheader("🔗 변수 간 상관관계 (Heatmap)")
            corr = df.select_dtypes(include=[np.number]).corr()
            fig_corr = obj.Figure(data=obj.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale='RdBu_r',
                zmin=-1, zmax=1
            ))
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)

    except Exception as e:
        st.error(f"❌ 오류 발생: {e}")
        st.info("구분자(Delimiter)나 건너뛸 행(Skip Rows) 설정을 확인해 주세요.")

else:
    # Welcom Page
    st.info("왼쪽 사이드바에서 파일을 업로드하여 분석을 시작하세요.")
    st.image("https://images.unsplash.com/photo-1551288049-bbda38a5f452?auto=format&fit=crop&q=80&w=1000", caption="R&D Data Visualization")
