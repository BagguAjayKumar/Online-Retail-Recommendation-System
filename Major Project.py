#!/usr/bin/env python
# coding: utf-8

# In[78]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import cosine_similarity


# In[3]:


data = pd.read_excel("D:/WORK SPACE/Data Science/OnlineRetail (1) (1).xlsx")


# In[4]:


data.info()


# In[5]:


data.shape


# In[6]:


data.describe()


# In[7]:


data.head()


# In[8]:


data["Description"] = data["Description"].str.strip()


# In[9]:


data.dropna(axis=0,subset=["InvoiceNo"],inplace=True)     #removes duplicate InvoiceNo


# In[10]:


data.dropna(axis=0,subset=["CustomerID"],inplace=True)


# In[11]:


data["InvoiceNo"] = data["InvoiceNo"].astype('str')


# In[12]:


data = data[~data['InvoiceNo'].str.contains('C')]        #remove the Credit transactions


# In[13]:


data.head()


# In[14]:


data.shape


# In[15]:


data["StockCode"].value_counts()


# In[16]:


data.dtypes


# In[17]:


data["Sales"] = data["Quantity"]*data["UnitPrice"]


# In[18]:


data


# In[19]:


data["Month"] = data["InvoiceDate"].dt.month


# In[20]:


data.head()


# In[21]:


data.tail()


# Globally Popular Products

# In[109]:


globally_popular = data.groupby('Description')['Quantity'].sum().sort_values(ascending=False)
globally_popular


# In[110]:


plt.figure(figsize=(12,6))
globally_popular.head(10).plot(kind='bar')
plt.title('Top 10 Globally Popular Products')
plt.xlabel('Product Description')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.show()
print("Top 10 Globally Popular Products:")
print(globally_popular.head(10))


# County-Wise Popular Products

# In[114]:


country_popular = data.groupby(['Country','Description'])['Quantity'].sum().reset_index()
country_popular = country_popular.sort_values(['Country','Quantity'], ascending=[True, False])
country_popular = country_popular.groupby('Country').first().reset_index()


# In[115]:


country_popular


# In[117]:


plt.figure(figsize=(12,6))
sns.barplot(x='Country',y='Quantity',data = country_popular.head(10))
plt.title("Most Popular Product in Top 10 Countries")
plt.xlabel('Country')
plt.ylabel('Quantity of most Popular Product')
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.show()
print("\nMost Popular Product in Each Country:")
print(country_popular[['Country','Description','Quantity']].head(10))


# Month-Wise Popular Products

# In[118]:


monthly_popular = data.groupby(['Month','Description'])['Quantity'].sum().reset_index()
monthly_popular = monthly_popular.sort_values(['Month','Quantity'],ascending=[True, False])
monthly_popular = monthly_popular.groupby('Month').first().reset_index()
monthly_popular


# In[119]:


plt.figure(figsize=(12,6))
sns.lineplot(x='Month', y = 'Quantity', data = monthly_popular)
plt.title("Quantity of Most Popular Product Each Month")
plt.xlabel("Month")
plt.ylabel("Quantity of Most Popular Product")
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.show()
print("\nMost Popular Product Each Month:")
print(monthly_popular[['Month','Description','Quantity']])


# In[ ]:





# In[120]:


data["Quantity"].sort_values(ascending=False)


# Most Popular Item globally is having InvoiceNo of 540421

# In[121]:


data['Country'].value_counts()


# United Kingdom is having more items transactions

# In[122]:


Pivot_table = data.pivot_table(index='CustomerID',columns='StockCode',values='Sales',aggfunc="sum",fill_value=0)
Pivot_table.head()


# In[123]:


correlation_matrix = Pivot_table.corr()


# In[124]:


plt.figure(figsize=(12,10))
sns.heatmap(correlation_matrix.iloc[:20,:20], annot=False,cmap="coolwarm")
plt.title("Product Correlation Heatmap of first 20 Products")
plt.show()


# In[125]:


def get_recommendations_with_prediction(CustomerID,StockCode,Pivot_table,correlation_matrix,n=5):
    if StockCode not in correlation_matrix.columns:
        return pd.Series()
    similar_products = correlation_matrix[StockCode].sort_values(ascending=False)
    similar_products.iloc[1:n+1]
    
    customer_purchases = Pivot_table.loc[CustomerID]
    
    predictions = []
    for prod in similar_products.index:
        if prod in customer_purchases.index:
            prediction = similar_products[prod]*customer_purchases[prod]
            predictions.append(prediction)
        else:
            predictions.append(0)
    recommendations = pd.Series(predictions, index=similar_products.index)
    recommmedations = recommendations.sort_values(ascending=False)
    return recommendations


# In[126]:


CustomerID = 17850.0
StockCode = '84029G'
recommendations = get_recommendations_with_prediction(CustomerID,StockCode,Pivot_table,correlation_matrix)
print(f"Recommendatios for customer {CustomerID} based on product {StockCode}:")
print(recommendations)


# In[127]:


data.head()


# In[128]:


def get_top_recommendations(CustomerID, Pivot_table, correlation_matrix, n=5):
    customer_purchases = Pivot_table.loc[CustomerID]
    purchased_products = customer_purchases[customer_purchases > 0].index
    
    all_recommendations = pd.Series()
    for product in purchased_products:
        recommendations = get_recommendations_with_prediction(CustomerID, product, Pivot_table, correlation_matrix)
        all_recommendations = all_recommendations.add(recommendations, fill_value=0)
    
    # Remove products the customer has already purchased
    all_recommendations = all_recommendations[~all_recommendations.index.isin(purchased_products)]
    
    return all_recommendations.sort_values(ascending=False).head(n)

# Example usage
customer_id = 13047.0
top_recommendations = get_top_recommendations(CustomerID, Pivot_table, correlation_matrix)
print(f"\nTop recommendations for customer {CustomerID}:")
print(top_recommendations)


# This model may not predict accurate because of its limited columns/data. The dataset provided does not contain information about customers which is crucial for personalized recommendations, no detailed product information is avaliable,customer behaviour is unknown and many more aspects.
# This prediction process in this system is based on Collaborative filtering, specifically item-based collaborative filtering, the main idea behind this is if a Customer likes a product then they also like a similar products.The similarity between products is determined by how often they are purchased together by all customers.
