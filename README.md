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
     


   ```
      from selenium import webdriver
      from datetime import datetime
      
      #dates of the html pages
      d_path = "data/html"
      save_path = "data/dates.txt"
      
      if not os.path.exists(save_path):
          data_dict=dict()
      
          with open(save_path, 'w') as f:
              f.write("id date\n")
      
              driver=webdriver.Chrome()
      
              if os.path.exists(d_path):
                  for file in os.listdir(d_path):
                      full_path = os.path.join(d_path, file)
      
                      try:
                          #open the file in the browser
                          driver.get('file://' + os.path.abspath(full_path))
                          html_source = driver.page_source
                          soup=BeautifulSoup(html_source, 'html.parser')
                          date=soup.find('div',{'class':'pt-3 text-base text-gray-400 sm:pt-4'}).text
                          
                          #if date is in format NAME•Month Day, Year
                          if '•' in date:
                              date=date.split('•')[1]
                              
                          date=datetime.strptime(date, '%B %d, %Y').date()
                          id=file.split('.')[0]
      
                          #to avoid running selenium every time, save the dates to a file
                          if date is not None and id is not None:
                              f.write(" ".join([id, str(date)]) + '\n')
                          data_dict[id]=date
                          
                      except:
                          pass
      
              driver.close()

2. Boosting Regression Model:
   - Initially used a Decision Tree
   - Experimented with other models like VotingRegressor, RandomForestRegressor, AdaBoostRegressor
   - Since the data was biased due to the uneven distribution among different grades, we transitioned to employing Gradient Boosting Regression (with a learning rate of 0.25), which is good for handling imbalanced datasets.


   ```
      grd3 = GradientBoostingRegressor(random_state=0,criterion='squared_error', learning_rate=0.25, n_estimators=49)
3. Feature Engineering:
   - Introduced Keywords2Search list for identifying the occurrence of specific keywords in user prompts
   - Introduced Keywords2SearchResponse for detecting the presence of keywords like 'python' 'code' and 'import' to check whether or not chatGPT's reponses were providing Python code and how many times code was provided.
   - The sum of average prompt length and average response length gives the total average length of a conversation.



   ```
      keywords2search = ["error", "no", "thank", "next", "help", "also", "explain"]
      keywords2search = [k.lower() for k in keywords2search]
      
      keywords2searchResponses=["python", "copy code", "import"]
      keywords2searchResponses = [k.lower() for k in keywords2searchResponses]
      
      for code, convs in code2convos.items():
          if len(convs) == 0:
              print(code)
              continue
          for c in convs:
              text = c["text"].lower()
              if c["role"] == "user":
                  # User Prompts
      
                  # count the user prompts
                  code2features[code]["#user_prompts"] += 1
                  
                  # count the keywords
                  for kw in keywords2search:
                      code2features[code][f"#{kw}"] +=  len(re.findall(rf"\b{kw}\b", text))
      
                  code2features[code]["prompt_avg_chars"] += len(text)
              else:
                  # ChatGPT Responses
                  code2features[code]["response_avg_chars"] += len(text)
                  for kw in keywords2searchResponses:
                      code2features[code][f"#{kw}"] +=  len(re.findall(rf"{kw}", text))
      
              code2features[code]["prompt_avg_chars"] /= code2features[code]["#user_prompts"]   
              code2features[code]["response_avg_chars"] /= code2features[code]["#user_prompts"]
              code2features[code]["total_avg_chars"] = code2features[code]["prompt_avg_chars"] + code2features[code]["response_avg_chars"]
## Results
![image](https://github.com/asmaabashir/CS412-ML/assets/127853761/d9fcfb25-6c84-442b-b599-d0da7e91226f)

As can be seen in the histogram above, the grades are heavily biased. Our dataset is also very small.
Both of these factors heavily affect the performance of our model.

We improved the performance of the model and raised our final test score to 0.57.

## Issues
1. Repeated Prompts Analysis:
   - Attempted to identify repeated prompts about the same homework question, but faced challenges and eventually excluded this feature from our model.

2. Sampling Methods:
   - Experimented with different sampling methods; attempted to use the RandomOverSampler from the imbalanced-learn library for oversampling the minority class, but ended up excluding it because it was adversely affecting the model due to the small size of the dataset.
  
## Contributions
Kaan Guray Sirin: Data Extraction/Preprocessing, Feature Engineering, Model Training

Asmaa Bashir: Feature Engineering, Model Evaluation, Documentation

Hannan Toprak: Feature Engineering, Model Evaluation, Documentation
