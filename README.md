# Heritage Housing Issues

Heritage Housing Issues is a data science and machine learning project with an end goal of estimating the sale price of inherited properties. We will use a Streamlit dashboard to meet the client's expectations. We will achieve this by enabling them to visualize which features of the dataset are most closely correlated to the property price. They will also be able to estimate the sale price of the inherited property, individually and as a whole. Finally the client (and/or future customers) to manually select a house's features and estimate it's sale price.

## Dataset Content

- This project uses a dataset sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data).
- The first part of the dataset, house_price_records, has 1460 rows and 24 columns. 23 of them represent the house profile from properties in Ames, Iowa, built between 1872 and 2010 (i.e: Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built). The last column represents the sale price for the property, this will be our target variable.

| Variable      | Meaning                                                                 | Units                                                                                                                                                                   |
| :------------ | :---------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1stFlrSF      | First Floor square feet                                                 | 334 - 4692                                                                                                                                                              |
| 2ndFlrSF      | Second-floor square feet                                                | 0 - 2065                                                                                                                                                                |
| BedroomAbvGr  | Bedrooms above grade (does NOT include basement bedrooms)               | 0 - 8                                                                                                                                                                   |
| BsmtExposure  | Refers to walkout or garden level walls                                 | Gd: Good Exposure; Av: Average Exposure; Mn: Minimum Exposure; No: No Exposure; None: No Basement                                                                       |
| BsmtFinType1  | Rating of basement finished area                                        | GLQ: Good Living Quarters; ALQ: Average Living Quarters; BLQ: Below Average Living Quarters; Rec: Average Rec Room; LwQ: Low Quality; Unf: Unfinshed; None: No Basement |
| BsmtFinSF1    | Type 1 finished square feet                                             | 0 - 5644                                                                                                                                                                |
| BsmtUnfSF     | Unfinished square feet of basement area                                 | 0 - 2336                                                                                                                                                                |
| TotalBsmtSF   | Total square feet of basement area                                      | 0 - 6110                                                                                                                                                                |
| GarageArea    | Size of garage in square feet                                           | 0 - 1418                                                                                                                                                                |
| GarageFinish  | Interior finish of the garage                                           | Fin: Finished; RFn: Rough Finished; Unf: Unfinished; None: No Garage                                                                                                    |
| GarageYrBlt   | Year garage was built                                                   | 1900 - 2010                                                                                                                                                             |
| GrLivArea     | Above grade (ground) living area square feet                            | 334 - 5642                                                                                                                                                              |
| KitchenQual   | Kitchen quality                                                         | Ex: Excellent; Gd: Good; TA: Typical/Average; Fa: Fair; Po: Poor                                                                                                        |
| LotArea       | Lot size in square feet                                                 | 1300 - 215245                                                                                                                                                           |
| LotFrontage   | Linear feet of street connected to property                             | 21 - 313                                                                                                                                                                |
| MasVnrArea    | Masonry veneer area in square feet                                      | 0 - 1600                                                                                                                                                                |
| EnclosedPorch | Enclosed porch area in square feet                                      | 0 - 286                                                                                                                                                                 |
| OpenPorchSF   | Open porch area in square feet                                          | 0 - 547                                                                                                                                                                 |
| OverallCond   | Rates the overall condition of the house                                | 10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor                                 |
| OverallQual   | Rates the overall material and finish of the house                      | 10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor                                 |
| WoodDeckSF    | Wood deck area in square feet                                           | 0 - 736                                                                                                                                                                 |
| YearBuilt     | Original construction date                                              | 1872 - 2010                                                                                                                                                             |
| YearRemodAdd  | Remodel date (same as construction date if no remodelling or additions) | 1950 - 2010                                                                                                                                                             |
| SalePrice     | Sale Price                                                              | 34900 - 755000                                                                                                                                                          |

- The second part of the dataset, inherited_houses, has 4 rows and 23 columns. Each row represents an inherited property and the columns correspond to the house profile. The variables are consistent in name and type with We will establish a ML process to determine the sales price for each property.

## Business Requirements

As a good friend, you are requested by your friend, who has received an inheritance from a deceased great-grandfather located in Ames, Iowa, to help in maximising the sales price for the inherited properties.

Although your friend has an excellent understanding of property prices in her own state and residential area, she fears that basing her estimates for property worth on her current knowledge might lead to inaccurate appraisals. What makes a house desirable and valuable where she comes from might not be the same in Ames, Iowa. She found a public dataset with house prices for Ames, Iowa, and will provide you with that.

- 1 - The client is interested in discovering how the house attributes correlate with the sale price. Therefore, the client expects data visualisations of the correlated variables against the sale price to show that.
- 2 - The client is interested in predicting the house sale price from her four inherited houses and any other house in Ames, Iowa.

## Hypothesis and how to validate?

- We will be making hypothesises regarding the sale price of houses:
  - The size of the house is positively correlated to the sale price. The bigger the house, the more expensive it is.
  - The quality/condition of the property is positively correlated to the sale price. A pretty house is an expensive house.
  - The age of a house is negatively correlated to the sale price. The older the house/renovation, the cheaper it is.

## The rationale to map the business requirements to the Data Visualisations and ML tasks

- Business requirement 1:

  - As a customer I can access an interactive dashboard, so that I can view and understand the data presented.
  - As a customer I can easily visualize correlation between variables, so that I can understand the impact of each feature on the sale price.
  - As a customer I can view the most influential features, so that I can concentrate on the right set of features.

- Business requirement 2:
  - As a customer I can visualize the predicted sales price of my inherited properties, so that I can predict my income.
  - As a customer I can enter a house's features and immediately predict it's sales price, so that I can predict if an investment is worth it.

## ML Business Case

- This project will enable the customer to predict sales price for houses in Ames, Iowa. This will allow them to make the correct investment choices and maximise their return.

- The customer has asked us to deliver the project through a dashboard. This dashboard will include different pages to meet the business requirements while maintaining an accessible yet comprehensive content.

- In order for the customer to be satisfied and consider a successful project outcome we will have to deliver a dashboard that is accessible and gives sufficient and clear insight on the house pricing. We will also have to be able to predict a house's price, based on certain features, with a minimum of 75% accuracy according to the model.

- The inputs for this model will be the houses' features. Proper study of the data will determine if all features are relevant, and with which level of correlation. The output will be a house price, in dollars. This price is the end target of our customer, who wants this prediction to be as precise as possible as they have inherited four houses, and might want to invest more. In order to meet the first goal, we will present the price of each inherited house as well as the total of all four houses. For the second goal, we will present the predicted sale price based on the selected features.

- Business requirement 1 considers visualizing data and correlation between features. As such it can be solved using conventional data analysis. In this case we will be studying correlations through Pearson's and Spearman's correlation analysises. Running a PPS study will help us understand how useful a feature is in predicting the value of our target variable by normalizing the data. We will also be using heatmaps and plots to understand the effects of a feature on our target variable.

- In this project business requirement 2 requires an artificial inteligence solution. We will train a ML model in order to achieve this.

- Our target variable is a price in dollars. As this is a discrete variable it suggests we should use a regression model. This will be a supervised and uni-dimensional Machine Learning task.

- The criteria for the performance goal of the predictions will be it's R2 score. The R2 score is the proportion of the variance in the dependant variable that is predictablefrom the independant variable. A high value will show a high level of correlation meaning our regression model is valid. The outcome will be considered as successful if the R2 score is of at least 0.75 on train and test sets.

- There are no ethical or privacy concerns with the data used in this project as it is a public dataset. This dataset contains house prices for Ames, Iowa. There are no names, addresses or any other personal/geographical information in the database. We will consider the area (62.86 km2 with 66427 inhabitants) wide enough to not cause any privacy issues.

- In order to tackle this complex project, we will divide it into epics and user stories. An epic is a body of work that can be broken down into specific tasks (called user stories) based on the needs/requests of customers or end-users. User stories are small, self-contained units of development work designed to accomplish a specific goal within our project. These will be used in conjunction with a Kanban board and MoSCoW prioritisation to ensure an efficient, focused and timely project delivery.

## Dashboard Design

- This project is presented via a StreamLit dashboard web app. It will allow the user to easily navigate through the pages via the interactive menu on the lefthand-side of the page.

### Main page - Project summary

- The main page presents a summary of the project. This is a brief description of the project's key terms, an overview of the dataset source and contents and an insight into our business requirements.

### Page two - Sales price study

- This page will display the results of our study of the dataset through correlation and PPS.
- It includes:
  - A sample view of our dataset
  - A graphical representation of our target variable
  - Graphical representations of the distribution of our target variable per feature
  - Heatmaps displaying correltion levels (Pearson and Spearman)
  - A heatmap showing the Predictive Power Score

### Page three - Project hypothesises

- Hypothesis one: the size of the house is positively correlated to the sale price.
- Validation: The correlation between features 1stFlrSF, GarageArea, GrLivArea, LotFrontage, TotalBsmtSF and SalePrice confirm this.

- Hypothesis two: the quality/condition of the property is positively correlated to the sale price.
- Validation: The correlation between features BsmtFinType1, GarageFinish,KitchenQual or OverallQual confirm this.

- Hypothesis three: the age of a house is negatively correlated to the sale price.
- Validation: The correlation between features GarageYrBlt, YearBuilt and YearRemodAdd confirm this.

### Page four - Inheritance sales price

- This page presents the prediction made for the four inherited houses. It allows the user to see the houses' most influencial features, as determined in our Modeling and Evaluation notebook. It also displays the individual predicted sales price as well as the total sales price expected.

### Page five - Sales price predictor

- This page gives the user the opportunity to input different values for a house's features and predict it's sales price in real time.

### Page six - Technical information

- This is the technical page of the project. It presents the results of our model's perfromance and the pipeline steps.

## Unfixed Bugs

- There are curently no known unfixed bugs.

## Deployment

### Heroku

- The App live link is:
- Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

- Here you should list the libraries you used in the project and provide example(s) of how you used these libraries.

## Credits

- In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism.
- You can break the credits section up into Content and Media, depending on what you have included in your project.

### Content

- The text for the Home page was taken from Wikipedia Article A
- Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

- The photos used on the home and sign-up page are from This Open Source site
- The images used for the gallery page were taken from this other open-source site

## Acknowledgements (optional)

- In case you would like to thank the people that provided support through this project.
