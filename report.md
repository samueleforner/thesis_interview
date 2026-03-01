# Report

### Samuele Forner
### March 1st - 2026

## The Task
Given a dataset representing car accidents in the USA, the request is to:
1) Train a model that is able to predict the level of severity of new unseen data (an integer value spanning from 1 to 4)

2) Find the factors that most influence the model predictions

Note_1: the subsampled version of the dataset was used (1milion rows)

Note_2: While K Folds CV would have been preferred for estimating the models performance, technical problems made necessary to resort to an Hold Out validation set.

## Data Preprocessing
### Imputation
The dataset presented missing values:

- The latitude / longitude where the accident ends:
    
    Since, in the rows where one of them was missing, the Distance between the start and end point was less or equal to 0.1 miles (93% of the cases) they have been replaced with the starting latitude / longitude

- The Weather Timestamps:

    It differs, on average, only of 27 seconds from the Start time timestamps, so the latter has been used to replace the missing values of the first one

- Some categorical attributes with high granularity have been removed entirely, like Streets, Cities, Counties and Zipcodes

- Other categorical attributes' missing values have been replaced with their mode

- Missing numerical attributes have been replaced with:
    
    - their mean, when their estimated distribution resambled a Gaussian or was centered around a specific point wirh low variance
    - their median, otherwise (to mitigate outliers effects)

### Data Encoding
- Timestamps have been replaced with hours, days of the week and months. This new features, being cyclic, have been mapped to their sines and cosines. Moreover, the accident duration has been computed from the difference between ending time and starting time of the accident
- One Hot Encoding has been applied on the State column, keeping only the 3 most frequent states and grouping the remaining under 'Other', a similar approach is used also for the Airport Code
- Before One Hot Encoding, duplicates are removed from Wind Direction
- Significant terms are extracted from the attributes 'Weather Conditions' 
- At first, the Description column were completely dropped. With later refinements the most frequent words were extracted from the columns and their presence in the description of each row has been one hot encoded. This alone resulted in an F1-score incrase from around 0.41 to 0.6

## Data Splitting
Attributes usefull for inference and targets are spearated, the the first ones are splitted in train, validation and test set.

The Description column has been dropped, but further improvement could make use of it, using an embedding system to extract useful informations

## Model Selection
Note that the targets were severely imbalanced towards class 2. This is the major factor that affected all the models' performances, to partially reduce this phenomenon, all the models, during traning, weight more errors caused by missclassification of under-represented classes.

Other techniques were taken into consideration, like SMOTE, that didn't work well due to the large presence of one hot encoded attributes (taking the convex combination of values designed to assume only binary values wasn't optimal) and undersampling (that would reduce the dataset drastically to match the minority class)

The preformance measure observed is the macro F1-score

- The first model considered is Logistic Regression, that performed already better than the dummy classifier (designed to predict the majority class), but still not over an F1-score of 40%. This was not surprising since this model assumes a linear relation between the input features and the log odds of the target, a strong hypothesis that is hard to satisfy with real data
- Random Forest (both Classifier and Regressor) yields the best results: bagging helped to reduce variance and trees are able to model non-linear relationships,  where the previous model failed
- Searching for better performance, Gradient Boosting was the next most promising option, both Hist Gradient Boosting and XGBoost have been tried, but without any sensible gain

As final model the Random Forest Regressor has been chosen instead of the correspective Classifier, but the performance are very similar

# Feature Importance

In a Random Forest model the importance that a feature had in the predictions can be estimated as the mean reduction of impurity in the dataset when the trees split on that attribute. For our model the most important attributes result to be the presence of the word 'closed' and 'exit' in the description, the distance ran by the vehicle, the presence of the word 'lane', the duration of the accident in seconds. Less important but still not negligible are the precipitations and wind temperature. It's interesting to notice that a similar importance can be observed for the month when the accident was recorded, the most plausible theory is that this is directly linked to the typical weather. 

Further analysis using Shapley Additive Explanations (SHAP) provided similar results
