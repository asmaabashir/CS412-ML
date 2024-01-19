# CS412 - HW Grade Predictor

This project aims to predict the score of a ML homework assignment by analyzing chatGPT histories submitted by the students. Initially utilizing a Decision Tree, the project transitioned to Gradient Boosting Regression due to heavy bias in the data. Key features include feature engineering, date extraction from HTML, and efforts to improve model performance.

## Table of Contents
- [Methodology](#methodology)
- [Results](#results)
- [Issues](#issues)
- [Contributions](#contributions)

## Methodology
1. Date Scraping & Extraction:
   - Used Selenium WebDriver to extract dates from HTML pages.
   - It iterates through each HTML file, extracts the date information using BeautifulSoup after rendering the page, and standardizes the date format to 'Month Day, Year.'
   - Resulting dates are then saved in a dates.txt file, establishing a mapping between each homework assignment ID and its corresponding date.
   - This allows us to explore potential correlations between submission times and final scores.
     
2. Boosting Regression Model:
   - Initially used a Decision Tree
   - Experimented with other models like VotingRegressor, RandomForestRegressor, AdaBoostRegressor
   - Since the data was biased due to the uneven distribution among different grades, we transitioned to employing Gradient Boosting Regression (with a learning rate of 0.25), which is good for handling imbalanced datasets.
     
3. Feature Engineering:
   - Introduced Keywords2Search list for identifying the occurrence of specific keywords in user prompts
   - Introduced Keywords2SearchResponse for detecting the presence of keywords like 'python' 'code' and 'import' to check whether or not chatGPT's reponses were providing Python code and how many times code was provided.
   - The sum of average prompt length and average response length gives the total average length of a conversation.

## Results
![image](https://github.com/asmaabashir/CS412-ML/assets/127853761/d9fcfb25-6c84-442b-b599-d0da7e91226f)

As can be seen in the histogram above, the grades are heavily biased. Our dataset is also very small.
Both of these factors heavily affect the performance of our model.
We improved the performance of the model and raised our final test score to 0.57.

## Issues
1. Repeated Prompts Analysis:
   - Attempted to identify repeated prompts about the same homework question, but faced challenges and eventually excluded this feature from our model.

2. Sampling Methods:
   - Experimented with different sampling methods; attempted to use the RandomOverSampler from the imbalanced-learn library for oversampling the minority class, but ended up exclusding it because it was aversely affecting the model due to the small size of the dataset.
  
## Contributions
Kaan: Data Extraction/Preprocessing, Feature Engineering, Model Training
Asmaa: Feature Engineering, Model Evaluation, Documentation
Hannan: Feature Engineering, Model Evaluation, Documentation
