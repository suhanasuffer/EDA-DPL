import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
import streamlit as st

df=pd.read_csv(r"puneProp.csv")
df['number of bedrooms'] = df['number of bedrooms'].astype(str)
my_list=df['number of bedrooms'].tolist()
input_values = my_list
numeric_results = []
for input_value in input_values:
    if not isinstance(input_value, str):
        continue
    numeric_part = re.findall(r'\d+', input_value)
    numeric_result = ''.join(numeric_part)
    numeric_result = int(numeric_result) if numeric_result else None
    numeric_results.append(numeric_result)
print(numeric_results)
my_list=numeric_results
df['number of bedrooms']=my_list
df.rename(columns={'val 2':'Price', 'val 4':'Status', 'area': 'Area (/sq ft)'}, inplace=True)
df['Area (/sq ft)'] = df['Area (/sq ft)'].str.replace(r'[^0-9]', '', regex=True).str.strip().str.replace(',', '', regex=True).astype(int)
df.dropna(inplace=True)
df.isna().sum().sum()
df['Price'] = df.apply(lambda row: (row['Price'] * 100) if row['unit'] == 'Cr' else row['Price'],axis=1)
df.drop('unit', axis=1, inplace=True)

st.title("Property Recommender")
user_budget = st.number_input("Enter your budget (in L): ")
user_area_per_sqft = st.number_input("Enter your required area per sq.ft: ")

if st.button("Recommend Properties"):
    filtered_data = df[(df['Price'] <= user_budget) & (df['Area (/sq ft)'] >= user_area_per_sqft)]
    df['Similarity'] = cosine_similarity(
        df[['Price', 'Area (/sq ft)']],
        [[user_budget, user_area_per_sqft]]
    )
    sorted_data = df.sort_values(by='Similarity', ascending=False)
    top_recommendations = sorted_data[['location', 'property_type', 'projName', 'Price', 'Area (/sq ft)']].head(10)

    st.subheader("Top Recommendations:")
    st.write(top_recommendations)
