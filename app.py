import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

st.set_page_config(page_title="Food Delivery EDA Dashboard", layout="wide")
st.title("🚚 Food Delivery Delay Analysis")

# -----------------------------
# Load & Clean Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")

    df['Delivery_person_Age'] = pd.to_numeric(df['Delivery_person_Age'], errors='coerce')
    df['Delivery_person_Ratings'] = pd.to_numeric(df['Delivery_person_Ratings'], errors='coerce')
    df['multiple_deliveries'] = pd.to_numeric(df['multiple_deliveries'], errors='coerce')

    df['Time_taken(min)'] = df['Time_taken(min)'].str.replace('(min) ', '', regex=False).astype(int)

    df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y')
    df['Order_Hour'] = pd.to_datetime(df['Time_Ordered'], format='%H:%M:%S', errors='coerce').dt.hour

    df['Weatherconditions'] = df['Weatherconditions'].str.replace('conditions ', '')

    return df

df = load_data()

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("🔎 Filters")
city = st.sidebar.multiselect("City", df['City'].unique(), default=df['City'].unique())
traffic = st.sidebar.multiselect("Traffic Density", df['Road_traffic_density'].unique(),
                                 default=df['Road_traffic_density'].unique())

df = df[(df['City'].isin(city)) & (df['Road_traffic_density'].isin(traffic))]

# -----------------------------
# KPI Metrics
# -----------------------------
c1, c2, c3 = st.columns(3)
c1.metric("⏱ Avg Delivery Time", f"{df['Time_taken(min)'].mean():.2f} min")
c2.metric("⭐ Avg Rating", f"{df['Delivery_person_Ratings'].mean():.2f}")
c3.metric("📦 Orders", df.shape[0])

# -----------------------------
# VISUALIZATIONS
# -----------------------------

# 1️⃣ Histogram
st.subheader("1️⃣ Distribution of Delivery Time")
fig, ax = plt.subplots()
ax.hist(df['Time_taken(min)'], bins=30)
ax.set_xlabel("Time (min)")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# 2️⃣ Box Plot
st.subheader("2️⃣ Traffic Density vs Delivery Time (Box Plot)")
fig, ax = plt.subplots()
sns.boxplot(x='Road_traffic_density', y='Time_taken(min)', data=df, ax=ax)
st.pyplot(fig)

# 3️⃣ Violin Plot
st.subheader("3️⃣ Weather Impact on Delivery Time (Violin Plot)")
fig, ax = plt.subplots()
sns.violinplot(x='Weatherconditions', y='Time_taken(min)', data=df, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# 4️⃣ Line Plot
st.subheader("4️⃣ Orders by Hour (Trend Analysis)")
hourly = df.groupby('Order_Hour').size()
fig, ax = plt.subplots()
ax.plot(hourly.index, hourly.values, marker='o')
ax.set_xlabel("Hour")
ax.set_ylabel("Number of Orders")
st.pyplot(fig)

# 5️⃣ Scatter Plot
st.subheader("5️⃣ Driver Rating vs Delivery Time (Relationship)")
fig, ax = plt.subplots()
sns.scatterplot(x='Delivery_person_Ratings', y='Time_taken(min)', data=df, ax=ax)
st.pyplot(fig)

# 6️⃣ Heatmap
st.subheader("6️⃣ Correlation Heatmap")
num_df = df[['Delivery_person_Age','Delivery_person_Ratings',
             'Vehicle_condition','multiple_deliveries','Time_taken(min)']]
corr = num_df.corr()

fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# 7️⃣ Bar Plot
st.subheader("7️⃣ Vehicle Type vs Avg Delivery Time")
vehicle_avg = df.groupby('Type_of_vehicle')['Time_taken(min)'].mean().sort_values()
fig, ax = plt.subplots()
vehicle_avg.plot(kind='bar', ax=ax)
ax.set_ylabel("Avg Time (min)")
st.pyplot(fig)

# 8️⃣ Count Plot
st.subheader("8️⃣ Order Type Frequency")
fig, ax = plt.subplots()
sns.countplot(x='Type_of_order', data=df, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# 9️⃣ Pie Chart
st.subheader("9️⃣ Festival Orders Proportion")
festival_counts = df['Festival'].value_counts()
fig, ax = plt.subplots()
ax.pie(festival_counts, labels=festival_counts.index, autopct='%1.1f%%')
st.pyplot(fig)

st.success("✅ EDA completed with multiple visualization techniques")
