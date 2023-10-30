import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv(r'puneProp.csv')
df.head()

df.columns
len(df.columns)

print("Size of dataframe: ", df.shape)

df.describe()
df.info()

df.isna().sum()
df.isna().sum().sum()

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
df.head()
df['Area (/sq ft)'] = df['Area (/sq ft)'].str.replace(r'[^0-9]', '', regex=True).str.strip().str.replace(',', '', regex=True).astype(int)
df.dtypes

df.dropna(inplace=True)
df.isna().sum().sum()
df['unit'].value_counts()
df['Price'] = df.apply(lambda row: (row['Price'] * 100) if row['unit'] == 'Cr' else row['Price'],axis=1)
df.drop('unit', axis=1, inplace=True)
df.head()

feature_to_analyze = 'property_type'
avg_prices = df.groupby(feature_to_analyze)['Price'].mean()

plt.figure(figsize=(20, 6))
avg_prices.plot(kind='bar')
plt.title(f'Real Estate Market Trends by {feature_to_analyze}')
plt.xlabel('Property type')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

avg_prices = df.groupby(feature_to_analyze)['Area (/sq ft)'].mean()

plt.figure(figsize=(20, 6))
avg_prices.plot(kind='bar')
plt.title(f'Real Estate Market Trends by property types')
plt.xlabel('Property type')
plt.ylabel('Area (/sq ft)')
plt.xticks(rotation=45)
plt.grid(axis='y')

plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(data=df, x='Price', color='lightblue')
plt.title('Price Distribution (Box Plot)')

plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='Area (/sq ft)', color='lightcoral')
plt.title('Area Distribution (Box Plot)')

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Area (/sq ft)', y='Price', color='lightseagreen')
plt.title('Scatter Plot: Price vs. Area')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price')
plt.show()

top_seller_names = df['seller-name'].value_counts().head(10).index
top_locations = df['location'].value_counts().head(10).index

filtered_data = df[(df['seller-name'].isin(top_seller_names)) & (df['location'].isin(top_locations))]

pivot_table = filtered_data.pivot_table(index='seller-name', columns='location', values='Price', aggfunc='count')

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True)
plt.title('Frequency of Sellers vs. Location')
plt.show()

seller_type_distribution = df["seller-name"].value_counts().reset_index()

plt.figure(figsize=(40, 4))
x = seller_type_distribution["index"]
y = seller_type_distribution["seller-name"]
sns.barplot(x=x, y=y, palette="Set2")
plt.xlabel("Seller Type")
plt.ylabel("Count")
plt.title("Distribution of Seller Types")
plt.xticks(rotation=90)
plt.show()

N = 50
top_sellers = df['seller-name'].value_counts().head(N)

plt.figure(figsize=(20, 6))
sns.barplot(x=top_sellers.index, y=top_sellers.values, palette="Set1")
plt.xlabel("Seller Name")
plt.ylabel("Frequency")
plt.title(f"Top {N} Sellers by Frequency")
plt.xticks(rotation=90)
plt.show()

numerical_columns = ["Price", "Area (/sq ft)", "number of bedrooms"]

sns.set()
sns.pairplot(df[numerical_columns], kind="scatter", diag_kind="kde")
plt.show()

user_budget = float(input("Enter your budget (in INR): "))
user_area_per_sqft = float(input("Enter your required area per sq.ft: "))

filtered_data = df[(df['Price'] <= user_budget) & (df['Area (/sq ft)'] >= user_area_per_sqft)]

df['Similarity'] = cosine_similarity(
    df[['Price', 'Area (/sq ft)']],
    [[user_budget, user_area_per_sqft]]
)

sorted_data = df.sort_values(by='Similarity', ascending=False)
top_recommendations = sorted_data[['location', 'property_type', 'projName', 'Price', 'Area (/sq ft)']].head(10)
print("\nTop Recommendations:")
print(top_recommendations)
