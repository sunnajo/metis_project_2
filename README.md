## **Predicting User Score/Rating of a Hip Hop Album**

*Description:*

Predict user score/rating of a hip hop album based on various features of the album and its artist using a linear regression model



*Variables:*

- Features: see summary of process/workflow below
- Target: aggregate user score for album (out of 100)



*Data Used:*

- Album of the Year: "The Best Hip Hop Albums of All Time by User Score" list, individual album pages, individual artist pages
  https://www.albumoftheyear.org/ratings/user-highest-rated/all/hip-hop/
- Spotify API: additional features about albums and artists, including Spotify popularity



*Tools Used:*

- Pandas
- Numpy
- Web scraping tools: BeautifulSoup
- Plotting modules: matplotlib, seaborn
- Regression modules: scikit-learn, statsmodels
- Spotipy python library to access Spotify API



*Summary of Process/Workflow:*

I used an iterative design process. I started out by testing a OLS linear regression model with features that had a correlation with the target variable of at least 0.2 with statsmodels. This model performed poorly (maximum R^2 score <0.5, high multi-collinearity, high complexity) and had several features that were not statistically significant. These features were re-evaluated and engineered (e.g. log or quadratic transformation, combined with co-linear features), the model was re-run, and the majority of these features were ultimately eliminated as they did not improve the model and remained statistically insignificant. Once a set of features that were statistically significant and did not appear obviously/highly colinear with a OLS R^2 score of >0.6 was obtained, linear regression models with and without regularization and polynomial regression models with and without regularization were validated and evaluated through calculating various metrics, such as the R^2 scores of the model run on both the training and validation sets, the training/validation R^2 score ratios, mean MAEs, mean RMSEs. Through regularization, a few variables were removed with improvement in the model. The final model was a degree 2 polynomial regression model with the following six features: number of user ratings of the album on AOTY, critic score of the album on AOTY, years since release of the album, whether or not the album was of the trap rap genre, whether or not the album was of the pop rap genre, the popularity of the artist on Spotify.



*Summary of Results, Possible Impacts:*

- The degree 2 polynomial regression model had a fairly good fit, with R^2 score 0.667 and low MAE/RMSE. Our predicted values for the target variable also had a linear correlation with the actual values.
- The model appeared to perform better with higher user scores. The residuals also had less variance at higher user scores. This may be due to the slightly skewed distribution of the target variable with higher user scores being more common than lower scores and/or the presence of low outliers.
- Looking at the coefficients of the features in the final model, it appears that the number of user ratings for the album, years since release of the album, and critic score of the album are positively correlated with user score, while the trap rap and pop rap genres and the popularity of the artist on Spotify are negatively correlated. The coefficient for the number of user ratings feature was relatively small but the model performed better when this variable was included, suggesting an interaction.
- While these relations are not definitive, these observations may be able to inform decisions about analysis of trends in the hip hop industry/consumption and future production of hip hop albums. It would be interesting to further analyze these correlations, especially why artists who are popular on Spotify produce albums that are rated less favorably by users, as well as to further analyze the outliers and possible interactions.
