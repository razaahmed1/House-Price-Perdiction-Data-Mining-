import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------
# Streamlit Config
# ----------------------------------
st.set_page_config(page_title="House Price Prediction", layout="wide")
st.title("🏠 House Price Prediction")

# ----------------------------------
# Load Data
# ----------------------------------
df = pd.read_csv("data.csv")

# ----------------------------------
# Feature Engineering
# ----------------------------------
CURRENT_YEAR = 2024

df["house_age"] = CURRENT_YEAR - df["yr_built"]
df["renovated"] = (df["yr_renovated"] > 0).astype(int)

# Log transform target (reduces error)
df["price_log"] = np.log1p(df["price"])

# ----------------------------------
# Remove Outliers (important)
# ----------------------------------
low = df["price_log"].quantile(0.05)
high = df["price_log"].quantile(0.95)
df = df[(df["price_log"] >= low) & (df["price_log"] <= high)]

# ----------------------------------
# Prepare Features & Target
# ----------------------------------
X = df.drop(["price", "price_log"], axis=1).select_dtypes(include=[np.number])
y = df["price_log"]

# ----------------------------------
# 80% TRAIN – 20% TEST SPLIT
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,   # 20% test
    random_state=42
)

# ----------------------------------
# Train Model (Cached)
# ----------------------------------
@st.cache_resource
def train_model(_X_train, _y_train):
    model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        min_samples_split=10,
        subsample=0.8,
        random_state=42
    )
    model.fit(_X_train, _y_train)
    return model

model = train_model(X_train, y_train)

# ----------------------------------
# Sidebar Navigation
# ----------------------------------
section = st.sidebar.radio(
    "Navigation",
    ["EDA", "Model Training", "Model Evaluation", "Prediction"]
)

# ----------------------------------
# EDA SECTION
# ----------------------------------
if section == "EDA":
    st.header("📊 Exploratory Data Analysis")

    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["price"], kde=True, ax=ax, color='skyblue')
        ax.set_xlabel("Price ($)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    with col2:
        st.subheader("Price vs Living Space")
        fig, ax = plt.subplots()
        sns.scatterplot(x="sqft_living", y="price", data=df, ax=ax, alpha=0.5)
        ax.set_xlabel("Living Space (sq ft)")
        ax.set_ylabel("Price ($)")
        st.pyplot(fig)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Price by Bedrooms")
        fig, ax = plt.subplots()
        sns.boxplot(x="bedrooms", y="price", data=df[df["bedrooms"] <= 6], ax=ax)
        ax.set_xlabel("Number of Bedrooms")
        ax.set_ylabel("Price ($)")
        st.pyplot(fig)
    
    with col4:
        st.subheader("Waterfront Impact")
        fig, ax = plt.subplots()
        sns.violinplot(x="waterfront", y="price", data=df, ax=ax)
        ax.set_xlabel("Waterfront (0=No, 1=Yes)")
        ax.set_ylabel("Price ($)")
        st.pyplot(fig)
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = df[["price", "sqft_living", "bedrooms", "bathrooms", "floors", "waterfront", "view", "condition"]].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax, fmt='.2f')
        st.pyplot(fig)
    
    with col6:
        st.subheader("House Condition Distribution")
        fig, ax = plt.subplots()
        df["condition"].value_counts().sort_index().plot(kind='bar', ax=ax, color='coral')
        ax.set_xlabel("Condition (1-5)")
        ax.set_ylabel("Count")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        st.pyplot(fig)

# ----------------------------------
# MODEL TRAINING
# ----------------------------------
elif section == "Model Training":
    st.header("🤖 Model Training")

    st.success("Gradient Boosting model trained using 80% training data.")

    st.markdown("""
    **Why Gradient Boosting?**
    - Captures non-linear relationships
    - Robust to outliers
    - Excellent for tabular datasets
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Importance")
        fig, ax = plt.subplots(figsize=(8, 6))
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        sns.barplot(data=feature_importance, y='feature', x='importance', ax=ax, palette='viridis')
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Features")
        ax.set_title("Top 10 Most Important Features")
        st.pyplot(fig)
    
    with col2:
        st.subheader("Training Data Split")
        fig, ax = plt.subplots()
        sizes = [len(X_train), len(X_test)]
        labels = [f'Training\n({len(X_train)} samples)', f'Testing\n({len(X_test)} samples)']
        colors = ['#ff9999', '#66b3ff']
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title("80-20 Train-Test Split")
        st.pyplot(fig)
    
    st.subheader("Model Hyperparameters")
    params_df = pd.DataFrame({
        'Parameter': ['n_estimators', 'learning_rate', 'max_depth', 'min_samples_split', 'subsample'],
        'Value': [500, 0.03, 4, 10, 0.8]
    })
    st.table(params_df)

# ----------------------------------
# MODEL EVALUATION
# ----------------------------------
elif section == "Model Evaluation":
    st.header("📈 Model Evaluation")

    y_pred_log = model.predict(X_test)

    y_pred = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test)

    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    r2 = r2_score(y_test_actual, y_pred)
    relative_error = (mae / y_test_actual.mean()) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"${mae:,.0f}")
    col2.metric("RMSE", f"${rmse:,.0f}")
    col3.metric("R² Score", f"{r2:.3f}")
    col4.metric("Relative Error", f"{relative_error:.2f}%")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Actual vs Predicted Prices")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test_actual, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
        
        # Add diagonal line for perfect predictions
        min_val = min(y_test_actual.min(), y_pred.min())
        max_val = max(y_test_actual.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel("Actual Price ($)")
        ax.set_ylabel("Predicted Price ($)")
        ax.set_title("Prediction Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Residuals Distribution")
        residuals = y_test_actual - y_pred
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(residuals, kde=True, ax=ax, color='purple')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.set_xlabel("Residuals (Actual - Predicted)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Prediction Errors")
        ax.legend()
        st.pyplot(fig)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Residuals vs Predicted")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel("Predicted Price ($)")
        ax.set_ylabel("Residuals ($)")
        ax.set_title("Residual Plot")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col4:
        st.subheader("Prediction Error Analysis")
        fig, ax = plt.subplots(figsize=(8, 6))
        percentage_error = np.abs((y_test_actual - y_pred) / y_test_actual * 100)
        
        bins = [0, 5, 10, 15, 20, 100]
        labels = ['0-5%', '5-10%', '10-15%', '15-20%', '>20%']
        error_categories = pd.cut(percentage_error, bins=bins, labels=labels)
        error_counts = error_categories.value_counts().sort_index()
        
        colors_palette = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c', '#c0392b']
        error_counts.plot(kind='bar', ax=ax, color=colors_palette)
        ax.set_xlabel("Error Range")
        ax.set_ylabel("Number of Predictions")
        ax.set_title("Prediction Error Distribution")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)

# ----------------------------------
# PREDICTION SECTION
# ----------------------------------
elif section == "Prediction":
    st.header("🔮 House Price Prediction")
    
    # Display model performance at the top
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test)
    mae = mean_absolute_error(y_test_actual, y_pred)
    r2 = r2_score(y_test_actual, y_pred)
    
    st.info(f"📊 Model Accuracy: R² = {r2:.3f} | Average Error: ±${mae:,.0f}")
    
    st.markdown("---")
    st.subheader("🏠 Enter Property Details")
    
    # Basic Property Information
    st.markdown("### 📐 Basic Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bedrooms = st.number_input("🛏️ Bedrooms", 1, 10, 3, help="Number of bedrooms")
    with col2:
        bathrooms = st.number_input("🚿 Bathrooms", 1.0, 10.0, 1.5, 0.5, help="Number of bathrooms")
    with col3:
        floors = st.number_input("🏢 Floors", 1.0, 4.0, 1.5, 0.5, help="Number of floors")
    
    # Size & Space
    st.markdown("### 📏 Size & Space")
    col4, col5 = st.columns(2)
    
    with col4:
        sqft_living = st.number_input("🏡 Living Area (sq ft)", 500, 13000, 1340, help="Interior living space")
    with col5:
        sqft_lot = st.number_input("🌳 Lot Size (sq ft)", 1000, 1000000, 7912, help="Total land area")
    
    col6, col7 = st.columns(2)
    
    with col6:
        sqft_above = st.number_input("⬆️ Above Ground (sq ft)", 500, 13000, 1340, help="Living space above ground")
    with col7:
        sqft_basement = st.number_input("⬇️ Basement (sq ft)", 0, 5000, 0, help="Basement area")
    
    # Property Features
    st.markdown("### ⭐ Property Features")
    col8, col9, col10 = st.columns(3)
    
    with col8:
        waterfront = st.selectbox("🌊 Waterfront", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Waterfront property")
    with col9:
        view = st.number_input("👁️ View Quality (0-4)", 0, 4, 0, help="Quality of view (0-4)")
    with col10:
        condition = st.number_input("🔧 Condition (1-5)", 1, 5, 3, help="Property condition (1-5)")
    
    # Age & History
    st.markdown("### 📅 Age & History")
    col11, col12 = st.columns(2)
    
    with col11:
        yr_built = st.number_input("🏗️ Year Built", 1900, 2024, 1955, help="Year of construction")
    with col12:
        yr_renovated = st.number_input("🔨 Year Renovated", 0, 2024, 0, help="Year of renovation (0 if never)")

    house_age = CURRENT_YEAR - yr_built
    renovated = int(yr_renovated > 0)
    
    st.markdown("---")
    
    # Predict Button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🔮 Predict House Price", use_container_width=True, type="primary")

    if predict_button:
        with st.spinner("Calculating prediction..."):
            input_dict = {
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "sqft_living": sqft_living,
                "sqft_lot": sqft_lot,
                "floors": floors,
                "waterfront": waterfront,
                "view": view,
                "condition": condition,
                "sqft_above": sqft_above,
                "sqft_basement": sqft_basement,
                "yr_built": yr_built,
                "yr_renovated": yr_renovated,
                "house_age": house_age,
                "renovated": renovated
            }

            input_df = pd.DataFrame([input_dict])
            input_df = input_df[X.columns]   # column alignment

            pred_log = model.predict(input_df)
            prediction = np.expm1(pred_log)
            
            # Display results with styling
            st.markdown("---")
            st.markdown("### 💰 Prediction Results")
            
            col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
            
            with col_res2:
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 30px; 
                                border-radius: 15px; 
                                text-align: center;
                                box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        <h1 style="color: white; margin: 0; font-size: 48px;">💵 ${prediction[0]:,.0f}</h1>
                        <p style="color: #f0f0f0; margin-top: 10px; font-size: 18px;">Estimated House Price</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Confidence interval
                lower_bound = prediction[0] - mae
                upper_bound = prediction[0] + mae
                
                st.info(f"📊 **Price Range:** ${lower_bound:,.0f} - ${upper_bound:,.0f}")
                st.warning(f"⚠️ **Error Margin:** ±${mae:,.0f}")
                
                # Property summary
                with st.expander("📋 Property Summary"):
                    st.write(f"**Bedrooms:** {bedrooms} | **Bathrooms:** {bathrooms} | **Floors:** {floors}")
                    st.write(f"**Living Area:** {sqft_living:,} sq ft | **Lot Size:** {sqft_lot:,} sq ft")
                    st.write(f"**Waterfront:** {'Yes' if waterfront == 1 else 'No'} | **View:** {view}/4 | **Condition:** {condition}/5")
                    st.write(f"**Built:** {yr_built} | **Age:** {house_age} years | **Renovated:** {'Yes' if renovated else 'No'}")