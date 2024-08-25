###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# Business Problem
###################################################

# One of the most important problems in e-commerce is the correct calculation of the points given to the products after the sale.
# The solution to this problem means more customer satisfaction for the e-commerce site, product prominence for sellers and a smooth shopping experience for buyers.
# Another problem is the correct ordering of the comments given to the products.
# Since the prominence of misleading reviews will directly affect the sales of the product, it will cause both financial loss and customer loss.
# In solving these 2 basic problems, e-commerce sites and sellers will increase their sales while customers will complete their purchasing journey smoothly.


###################################################
# Dataset Story
###################################################

# This dataset of Amazon product data includes product categories and various metadata.
# The most reviewed product in the electronics category has user ratings and reviews.

# Variables:
# reviewerID: User ID
# asin: Product ID
# reviewerName: User Name
# helpful: Useful evaluation rating
# reviewText: Evaluation
# overall: Product rating
# summary: Evaluation summary
# unixReviewTime: Evaluation time
# reviewTime: Evaluation time Raw
# day_diff: Number of days since evaluation
# helpful_yes: Number of times the evaluation was found useful
# total_vote: Number of votes for evaluation

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import math
import scipy.stats as st
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)



###################################################
# DUTY 1: Calculate the Average Rating according to Current Comments and Compare it with the Existing Average Rating.
###################################################

# In the shared dataset, users have rated and commented on a product.
# Our aim in this task is to evaluate the scores by weighting them by date.
# The initial average score should be compared with the weighted score based on the date.


###################################################
# Step 1:  Read the dataset and calculate the average score of the product.
###################################################
df_ = pd.read_csv("/Users/macbook/Desktop/Miuul/3. hafta meausemremnt problems/ödevler /Rating Product&SortingReviewsinAmazon/amazon_review.csv")
df= df_.copy()
df.shape
df.head()
df["overall"].mean()   # 4.5875

###################################################
# Step 2: Calculate the Weighted Grade Point Average by Date.
###################################################
df["day_diff"].quantile([.25, .5, .75])

# Determination of time-based average weights
def time_based_weighted_average(dataframe, w1=50, w2=25, w3=15, w4=10):
    return dataframe.loc[dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.25), "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.25)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.50)), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.50)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.75)), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.75)), "overall"].mean() * w4 / 100

time_based_weighted_average(df)   # 4.6373
time_based_weighted_average(df, w1=28, w2=26, w3=24, w4=22)


df.loc[df["day_diff"] <= df["day_diff"].quantile(0.25), "overall"].mean()
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.25)) & (df["day_diff"] <= df["day_diff"].quantile(0.50)),"overall"].mean()
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.50)) & (df["day_diff"] <= df["day_diff"].quantile(0.75)), "overall"].mean()
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.75)), "overall"].mean()
df["day_diff"]


df.loc[df["day_diff"] <= df["day_diff"].quantile(0.25), "overall"].mean() * w1/100 \
    df.loc[(df["day_diff"] > df["day_diff"].quantile(0.25) )& (df["day_diff"] <= df["day_diff"].quantile(0.5)), "overall"].mean() * w2/100 \
    df.loc[(df["day_diff"] > df["day_diff"].quantile(0.50)) & (df["day_diff"] <= df["day_diff"].quantile(0.75)), "overall"].mean() * w3/100 \
    df.loc[df["day_diff"] > df["day_diff"].quantile(0.75) , "overall"].mean() * w4/100

def weighted_average_by_time(dataframe, w1 = 40, w2=30, w3= 20, w4 =10):
    return dataframe.loc[dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.25), "overall"].mean() * w1/100 + \
            dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.25) )& (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.5)), "overall"].mean() * w2/100 + \
            dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.50)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.75)), "overall"].mean() * w3/100 + \
            dataframe.loc[dataframe["day_diff"] > dataframe["day_diff"].quantile(0.75) , "overall"].mean() * w4/100
weighted_average_by_time(df)

df.loc[df["day_diff"] <= df["day_diff"].quantile(0.25), "overall"].mean()
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.25) )& (df["day_diff"] <= df["day_diff"].quantile(0.5)), "overall"].mean()
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.50)) & (df["day_diff"] <= df["day_diff"].quantile(0.75)), "overall"].mean()
df.loc[df["day_diff"] > df["day_diff"].quantile(0.75) , "overall"].mean()



###################################################
# DUTY 2: Determine the 20 Reviews that will be displayed on the Product Detail Page for the product.
###################################################


###################################################
# Step 1. Generate helpful_no Variable
###################################################

# Note:
# total_vote is the total number of up-down votes given to a comment.
# "up" means helpful.
# There is no helpful_no variable in the dataset, it needs to be generated from existing variables.

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]


###################################################
# Step 2. Calculate score_pos_neg_diff, score_average_rating and wilson_lower_bound Scores and Add to Data
###################################################
def score_pos_neg_diff(up, down):
    return up - down

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["score_pos_nef_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis=1)
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)



##################################################
# Step 3. 20 Identify the Commentary and Interpret the Results.
###################################################

df.sort_values("wilson_lower_bound", ascending= False).head(20)



























