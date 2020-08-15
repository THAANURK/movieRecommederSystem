#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 12:16:01 2020

This code will recommend movies based on the ratings given by similar users for other movie, the concepts we used is correlation.

 _____  _   _ ______  _____  _____   _____  _   _   ___    ___   _   _  _   _ 
/  ___|| | | || ___ \|  ___||  ___| |_   _|| | | | / _ \  / _ \ | \ | || | | |
\ `--. | |_| || |_/ /| |__  | |__     | |  | |_| |/ /_\ \/ /_\ \|  \| || | | |
 `--. \|  _  ||    / |  __| |  __|    | |  |  _  ||  _  ||  _  || . ` || | | |
/\__/ /| | | || |\ \ | |___ | |___    | |  | | | || | | || | | || |\  || |_| |
\____/ \_| |_/\_| \_|\____/ \____/    \_/  \_| |_/\_| |_/\_| |_/\_| \_/ \___/ 


"""

# Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Gather Data
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names)
movie_titles = pd.read_csv("Movie_Id_Titles")

print(df.head())
print(movie_titles.head())
 
# Merging movie data with movi titles based on the item_id
df = pd.merge(df,movie_titles,on='item_id')
print(df.head())

# Data Visualisation based on rating and count 
sns.set_style('white')

# high rated movies not based on ppl
print(df.groupby('title')['rating'].mean().sort_values(ascending=False).head())

#max participation in rating movie
print(df.groupby('title')['rating'].count().sort_values(ascending=False).head())

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
print(ratings.head())

ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
print(ratings.head())

# distribution of ratings
plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70)

# outlier and partciption histogram to understand the curve : Contains outliers and the gaussian curve.
# Understanding the curve : 3 is the max chosen and outliers are 1 and 5 . the rough graph simply represnts the gaussian

plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)

# jointplot for rating and number of ratings with scatter check via the cluster
# on seeing the graph : the participation is higher from in 0 -100 ppl and ratings are 2 - 4.
sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)

# Its high time to reccomend movies
# This matrix contains user id and how much they rated the movies, most movie valu will be empty, cuz the user may have not watched all movie
moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
print(moviemat.head())

# higher participation for rating the movie in higher to lower order
print(ratings.sort_values('num of ratings',ascending=False).head(10))

# Will be  working with 2 movies initially
starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']
print(starwars_user_ratings.head())

# using correlation between two pandas series to find the similarity
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

# Fitering data : Removing NAN value
corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)

# This list may contain movie with less number of 5* rating,

print(corr_starwars.sort_values('Correlation',ascending=False).head(10))

# Filtering the data by taking count 100+ people participation in the rating event 

#Combining the ratings data
corr_starwars = corr_starwars.join(ratings['num of ratings'])
print(corr_starwars.head())

# filtering based on amt of ratings
print(corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head())

# The final output is like for starwars movie the next recommended movie will be  Empire Strikes Back, Return of the Jedi.

""" 
Star Wars (1977)                                       1.000000             584
Empire Strikes Back, The (1980)                        0.748353             368
Return of the Jedi (1983)                              0.672556             507
Raiders of the Lost Ark (1981)                         0.536117             420
Austin Powers: International Man of Mystery (1997)     0.377433             130
"""