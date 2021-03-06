# Relevancy prediction of news articles using neural networks

In this project I am using a variety of techniques to find the best model to predict the 'relevance' of a news article to the US economy. I wanted to use this project to further explore my interest in natural language processing whilst developing my skills using some of the more advanced capabilities of the Keras platform. The dataset I am using 'Economic News Article Tone and Relevance' is from 'figure eight' where contributors have read snippets of news articles and noted if the article was relevant to the US economy. The dataset has two columns containing the article headline and the snippet from the body of the text. As my previous project involved predicting the sentiment of tweets, I thought it would be interesting using the text variable as a predictor, which is alot more text heavy. In addition, I was also keen to test if combining the headline and text variables would improve the models accuracy compared to just looking at the text variable. My instinct was that it would, as a headline often gives a good indication of the content of the article.

## Instructions

Project details: Python version 3.5

<b>Python code</b>

- main.ipynb (notebook report)

<b>Python modules</b>

- preprocdata.py

Code has been run. Find outputs in notebook cells.

Data taken from Data for Everyone: https://www.figure-eight.com/data-for-everyone/

<b>Packages used</b>

- Keras
- collections
- pandas
- numpy
- os
- matplotlib
- sklearn

