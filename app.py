import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="K-Means House Price Prediction App", layout="wide"
)

st.title("🏡 Ứng dụng Dự đoán Giá nhà & Phân cụm K-Means")
st.write(
    "Mô hình huấn luyện trên tập Train, tính khoảng cách tới các Tâm cụm (Centroids) và đánh giá chính xác trên tập Test."
)

# ---------------------------------------------------------
# Load dữ liệu từ Upload hoặc File tự động
# ---------------------------------------------------------
st.sidebar.header("⚙️ Cấu hình dữ liệu & Tham số")

uploaded_train = st.sidebar.file_uploader(
    "Tải lên file Train (.xlsx)", type=["xlsx"]
)
uploaded_test = st.sidebar.file_uploader(
    "Tải lên file Test (.xlsx)", type=["xlsx"]
)


@st.cache_data
def load_datasets(u_train, u_test):
    df_train, df_test = None, None

    # Tìm file Train
    if u_train is not None:
        df_train = pd.read_excel(u_train)
    else:
        for fname in ["nhom4_train_data.xlsx", "nhom4_train_data_2.xlsx"]:
            if os.path.exists(fname):
                df_train = pd.read_excel(fname)
                break

    # Tìm file Test
    if u_test is not None:
        df_test = pd.read_excel(u_test)
    else:
        for fname in ["nhom4_test_data.xlsx", "nhom4_test_data_2.xlsx"]:
            if os.path.exists(fname):
                df_test = pd.read_excel(fname)
                break

    return df_train, df_test


train_df, test_df = load_datasets(uploaded_train, uploaded_test)

if train_df is None or test_df is None:
    st.error(
        "Không tìm thấy file dữ liệu! Vui lòng tải file `nhom4_train_data.xlsx` và `nhom4_test_data.xlsx` lên ứng dụng."
    )
    st.stop()

features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "HouseAge",
]
target = "SalePrice"

k_clusters = st.sidebar.slider("Chọn số lượng cụm (K)", 2, 10, 3)

# ---------------------------------------------------------
# Huấn luyện K-Means & Tính toán khoảng cách
# ---------------------------------------------------------
X_train = train_df[features].values
y_train = train_df[target].values
X_test = test_df[features].values
y_test = test_df[target].values

# 1. Khởi tạo & Phân cụm tập Train
kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
train_df["Cluster"] = kmeans.fit_predict(X_train)
centroids = kmeans.cluster_centers_

# 2. Tính lại trung bình SalePrice theo từng cụm trên Train
cluster_means = train_df.groupby("Cluster")[target].mean().to_dict()

# 3. Tính khoảng cách từ từng mẫu trong tập Test đến các Centroids
test_eval = test_df.copy()
for i in range(k_clusters):
    test_eval[f"Dist_Centroid_{i}"] = np.linalg.norm(
        X_test - centroids[i], axis=1
    )

# 4. Gán cụm gần nhất & Dự đoán giá
test_eval["Assigned_Cluster"] = kmeans.predict(X_test)
test_eval["Predicted_SalePrice"] = test_eval["Assigned_Cluster"].map(
    cluster_means
)
test_eval["Absolute_Error"] = np.abs(
    test_eval[target] - test_eval["Predicted_SalePrice"]
)

# ---------------------------------------------------------
# Các Tab hiển thị chính
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📋 Dữ liệu & Tâm Cụm",
        "📈 Phương pháp Elbow & Biểu đồ Cụm",
        "📊 Đánh giá trên tập Test",
        "🔮 Dự đoán Nhà mới",
    ]
)

# TAB 1: DỮ LIỆU & TÂM CỤM
with tab1:
    st.subheader("1. Dữ liệu huấn luyện & Kiểm thử (Gốc)")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Tập Train ({train_df.shape[0]} mẫu)**")
        st.dataframe(train_df[features + [target]].head(10))
    with c2:
        st.write(f"**Tập Test ({test_df.shape[0]} mẫu)**")
        st.dataframe(test_df[features + [target]].head(10))

    st.subheader("2. Tâm các cụm (Centroids) & Giá trung bình từng cụm")
    centroid_df = pd.DataFrame(centroids, columns=features)
    centroid_df["SalePrice_Mean"] = [
        cluster_means[i] for i in range(k_clusters)
    ]
    centroid_df.index.name = "Cluster"
    st.dataframe(
        centroid_df.style.highlight_max(axis=0, color="lightgreen")
    )

    st.markdown("### 💡 Phân tích ý nghĩa các cụm (với K=3):")
    st.markdown("""
    - **Cụm 0 (Nhà mới, chất lượng cao):** `OverallQual` cao, `HouseAge` rất thấp (nhà mới xây), giá bán trung bình lớn nhất.
    - **Cụm 1 (Nhà bình dân):** `OverallQual` thấp hơn trung bình, diện tích và quy mô vừa phải.
    - **Cụm 2 (Nhà lâu năm, Garage rộng):** `OverallQual` tốt, sức chứa `GarageCars` lớn nhưng `HouseAge` cao (nhà cũ).
    """)

# TAB 2: ELBOW & BIỂU ĐỒ CỤM
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("1. Biểu đồ Elbow (WCSS theo K)")
        wcss = []
        K_range = range(1, 11)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_train)
            wcss.append(km.inertia_)

        fig_elbow, ax_elbow = plt.subplots(figsize=(6, 4.2))
        ax_elbow.plot(K_range, wcss, "bo-", linewidth=2, markersize=7)
        ax_elbow.axvline(
            x=3,
            color="red",
            linestyle="--",
            label="Điểm Elbow tối ưu (K=3)",
        )
        ax_elbow.set_xlabel("Số cụm K")
        ax_elbow.set_ylabel("WCSS")
        ax_elbow.set_title("Xác định điểm gãy Elbow")
        ax_elbow.legend()
        ax_elbow.grid(True)
        st.pyplot(fig_elbow)

    with col_b:
        st.subheader("2. Biểu đồ phân bố giá nhà (SalePrice)")
        fig_dist, ax_dist = plt.subplots(figsize=(6, 4.2))
        sns.histplot(
            train_df[target], kde=True, ax=ax_dist, color="steelblue"
        )
        ax_dist.set_title("Phân bố biến mục tiêu SalePrice")
        ax_dist.set_xlabel("SalePrice")
        ax_dist.set_ylabel("Tần suất")
        st.pyplot(fig_dist)

    st.subheader("3. Biểu đồ phân cụm dữ liệu & Tâm cụm Centroid (PCA 2D)")
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    centroids_pca = pca.transform(centroids)

    fig_pca, ax_pca = plt.subplots(figsize=(9, 4.8))
    scatter = ax_pca.scatter(
        X_train_pca[:, 0],
        X_train_pca[:, 1],
        c=train_df["Cluster"],
        cmap="tab10",
        alpha=0.6,
        s=45,
    )
    ax_pca.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        c="red",
        marker="X",
        s=200,
        linewidths=2,
        label="Tâm cụm (Centroids)",
    )
    ax_pca.set_title(
        "Trực quan hóa không gian cụm và các Tâm cụm Centroid"
    )
    ax_pca.set_xlabel("PCA Component 1")
    ax_pca.set_ylabel("PCA Component 2")
    ax_pca.legend()
    ax_pca.grid(True)
    st.pyplot(fig_pca)

# TAB 3: ĐÁNH GIÁ TRÊN TẬP TEST
with tab3:
    st.subheader("1. Chỉ số đánh giá mô hình trên tập Test")
    mae = mean_absolute_error(y_test, test_eval["Predicted_SalePrice"])
    rmse = np.sqrt(
        mean_squared_error(y_test, test_eval["Predicted_SalePrice"])
    )
    r2 = r2_score(y_test, test_eval["Predicted_SalePrice"])

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE (Sai số tuyệt đối trung bình)", f"${mae:,.2f}")
    m2.metric("RMSE (Căn sai số bình phương TB)", f"${rmse:,.2f}")
    m3.metric("R² Score (Hệ số xác định)", f"{r2:.4f}")

    st.subheader(
        "2. Bảng khoảng cách Centroid & Dự đoán đối chiếu trên tập Test"
    )
    cols_to_show = (
        features
        + [f"Dist_Centroid_{i}" for i in range(k_clusters)]
        + [
            "Assigned_Cluster",
            target,
            "Predicted_SalePrice",
            "Absolute_Error",
        ]
    )
    st.dataframe(test_eval[cols_to_show])

    st.subheader("3. Biểu đồ so sánh Giá thực tế vs Giá ước tính")
    fig_comp, ax_comp = plt.subplots(figsize=(9, 4.5))
    ax_comp.plot(
        range(len(test_eval)),
        test_eval[target],
        label="Giá thực tế (Actual)",
        color="blue",
        alpha=0.75,
    )
    ax_comp.plot(
        range(len(test_eval)),
        test_eval["Predicted_SalePrice"],
        label="Giá ước tính (Predicted)",
        color="orange",
        linestyle="--",
        linewidth=2,
    )
    ax_comp.set_title(
        "So sánh trực tiếp Giá nhà Thực tế và Giá nhà Ước tính"
    )
    ax_comp.set_xlabel("Chỉ số ngôi nhà trong tập Test")
    ax_comp.set_ylabel("Giá nhà (SalePrice)")
    ax_comp.legend()
    ax_comp.grid(True)
    st.pyplot(fig_comp)

# TAB 4: DỰ ĐOÁN NHÀ MỚI
with tab4:
    st.subheader("Nhập thông số ngôi nhà mới để dự đoán phân cụm & giá")
    g1, g2, g3 = st.columns(3)
    in_qual = g1.number_input(
        "OverallQual (Chất lượng)", value=0.0, step=0.1
    )
    in_area = g2.number_input(
        "GrLivArea (Diện tích sống)", value=0.0, step=0.1
    )
    in_cars = g3.number_input(
        "GarageCars (Sức chứa Garage)", value=0.0, step=0.1
    )

    g4, g5 = st.columns(2)
    in_bsmt = g4.number_input(
        "TotalBsmtSF (Diện tích tầng hầm)", value=0.0, step=0.1
    )
    in_age = g5.number_input(
        "HouseAge (Tuổi thọ nhà)", value=0.0, step=0.1
    )

    if st.button("Tính toán Khoảng cách & Dự đoán giá"):
        user_point = np.array([[in_qual, in_area, in_cars, in_bsmt, in_age]])

        dists = [
            np.linalg.norm(user_point - centroids[i])
            for i in range(k_clusters)
        ]
        assigned_c = int(np.argmin(dists))
        pred_price = cluster_means[assigned_c]

        st.subheader("Kết quả phân tích khoảng cách:")
        for i in range(k_clusters):
            st.write(
                f"- Khoảng cách tới Centroid {i}: **{dists[i]:.4f}**"
            )

        st.success(
            f"🏠 Ngôi nhà thuộc **Cụm {assigned_c}** (Khoảng cách ngắn nhất: {dists[assigned_c]:.4f})"
        )
        st.info(f"💵 Giá dự đoán ước tính: **${pred_price:,.2f}**")
