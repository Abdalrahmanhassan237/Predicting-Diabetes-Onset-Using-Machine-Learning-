# Predicting-Diabetes-Onset-Using-Machine-Learning-

![0_7p7zUmwZ02iokYmP](https://github.com/Abdalrahmanhassan237/Predicting-Diabetes-Onset-Using-Machine-Learning-/assets/158060043/06e988d0-de81-4a21-bb82-27d4738d8d91)


## Explanation for Learners

### Importing Libraries

We start by importing essential libraries for data science tasks.

- **numpy (np)**: This library provides powerful tools for working with numerical data, like arrays and matrices. It's the foundation for many data science calculations.
- **pandas (pd)**: This library is designed for data analysis and manipulation. It allows us to load data from various sources (CSV, Excel, etc.), organize it into DataFrames (tabular structures), and perform cleaning and transformations.
- **matplotlib.pyplot (plt)**: This library is the core of Python's plotting functionality. It lets us create various visualizations like scatter plots, line graphs, and histograms to explore relationships within our data.
- **seaborn (sns)**: Seaborn is built on top of matplotlib and provides a higher-level interface for creating attractive and informative statistical graphics. It simplifies generating common plots used in data exploration and analysis.

### Splitting Data (train_test_split)

The `train_test_split` function from `sklearn.model_selection` helps us divide our dataset into two parts: training data and testing data.

- **Training data**: This larger portion of the dataset is used to train the machine learning model. The model learns patterns and relationships from this data.
- **Testing data**: This unseen portion (usually around 20-30% of the dataset) is used to evaluate the model's performance on data it hasn't seen before. This helps prevent overfitting (the model memorizing training data instead of learning general patterns).

### Machine Learning Algorithms

We import algorithms from the `sklearn` library that will be used to build our machine learning models. These algorithms learn from the training data to make predictions on new, unseen data.

- **LogisticRegression**: This is a popular algorithm for classification tasks, where we predict one of two or more possible categories for a data point. It's a good starting point for many classification problems due to its interpretability and efficiency.
- **SVC (Support Vector Machine)**: SVMs are another powerful classification algorithm, especially effective for handling data that may not be linearly separable. They can also be used for regression tasks (predicting continuous values).

### Ensemble Learning Algorithms

These algorithms combine the predictions of multiple models (decision trees in this case) to often achieve better performance than a single model.

- **RandomForestClassifier**: This algorithm creates a forest of decision trees, each trained on a random subset of features and data points. The final prediction is based on the majority vote of these trees, making it robust to noise or outliers in the data.
- **GradientBoostingClassifier**: This algorithm builds an ensemble of decision trees sequentially, where each tree learns to improve upon the predictions of the previous ones. It can be very effective for complex classification problems.
- 
## second

We import several functions from the `sklearn.metrics` module to evaluate the performance of a machine learning model.

- **confusion_matrix**: This function allows us to create a confusion matrix, which is a table that shows the counts of true positives, true negatives, false positives, and false negatives. It helps us understand the performance of a classification model by visualizing the predictions of the model compared to the true labels.

- **accuracy_score**: This function calculates the accuracy of the model's predictions. Accuracy is the ratio of correctly predicted data points to the total number of data points. It provides a general measure of how well the model performs overall.

- **recall_score**: Also known as sensitivity or true positive rate, this function calculates the ratio of true positives to the sum of true positives and false negatives. It quantifies the model's ability to correctly identify positive instances out of all actual positive instances.

- **f1_score**: This function calculates the F1 score, which is the harmonic mean of precision and recall. It provides a balanced measure of a model's performance, taking into account both precision (the ratio of true positives to the sum of true positives and false positives) and recall.

These functions are commonly used to assess the performance of classification models.

## Third 
### Explore Data 📊

1. Set the plot style to 'fivethirtyeight' using `plt.style.use('fivethirtyeight')`. This step is optional and makes the plots visually appealing.

2. Read a CSV file named 'diabetes.csv' located at the path '/kaggle/input/diabetes-data-set/' using `pd.read_csv('/kaggle/input/diabetes-data-set/diabetes.csv')`. The data is assigned to the variable `data`. This assumes that the CSV file contains a diabetes dataset.

3. Display the first five rows of the dataset using `data.head()`. This shows the top five rows of the DataFrame `data`.

4. Get information about the dataset using `data.info()`. This provides a summary of the dataset's structure, including column names, data types, and non-null values.

## Fourth 
### Data Analysis Process 📈

### data.describe()

The code `data.describe()` generates descriptive statistics of the numerical columns in the `data` DataFrame. It provides information such as count, mean, standard deviation, minimum, quartiles, and maximum values for each numerical feature.

### data.duplicated().sum()

The code `data.duplicated().sum()` checks for duplicate rows in the `data` DataFrame and returns the total number of duplicated rows. It sums up the boolean values returned by the `duplicated()` method, where `True` represents a duplicate row.

### data.corr()

The code `data.corr()` calculates the correlation between numerical columns in the `data` DataFrame. It returns a correlation matrix, which shows the pairwise correlation coefficients between each pair of columns. The correlation coefficient ranges from -1 to 1, where 1 indicates a strong positive correlation, -1 indicates a strong negative correlation, and 0 indicates no correlation.

### sns.heatmap(data.corr(), annot=True, fmt='0.2f', linewidth=0.6)

The code `sns.heatmap(data.corr(), annot=True, fmt='0.2f', linewidth=0.6)` creates a heatmap using the Seaborn library. It visualizes the correlation matrix calculated previously. The `annot=True` parameter displays the correlation values on the heatmap, `fmt='0.2f'` formats the values as floating-point numbers with two decimal places, and `linewidth=0.6` adjusts the width of the heatmap's lines.

### sns.countplot(x='Outcome', data=data, palette=['g', 'r'])

The code `sns.countplot(x='Outcome', data=data, palette=['g', 'r'])` creates a countplot using Seaborn. It shows the count of each unique value in the 'Outcome' column of the `data` DataFrame. The `x='Outcome'` parameter specifies the column to count, `data=data` indicates the DataFrame to use, and `palette=['g', 'r']` sets the color palette for the plot.
```python 
import warnings
warnings.filterwarnings('ignore')
```
These lines of code import the warnings module and temporarily ignore warnings generated by other code. This can be useful when you want to suppress specific warning messages temporarily.

### sns.displot(data.BMI)

The code `sns.displot(data.BMI)` creates a distribution plot using Seaborn. It visualizes the distribution of values in the 'BMI' column of the `data` DataFrame.

### sns.boxplot(data.Age)

The code `sns.boxplot(data.Age)` creates a boxplot using Seaborn. It displays the distribution of values in the 'Age' column of the `data` DataFrame, showing the quartiles, median, and potential outliers.

## Model Creation


### Separate features (X) and target variable (y)

```python
x = data.drop('Outcome', axis=1)  # Features for model training
y = data['Outcome']              # Target variable for prediction 
```

###  splits the data into training and testing sets

The provided code splits the data into training and testing sets, which is recommended for evaluating machine learning models.
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 
```

###  function evaluates the performance of a machine learning model



The `model_cal` function evaluates the performance of a machine learning model. Here's a step-by-step explanation of the code:


```python
def model_cal(model):
    # Print model information
    print("Evaluating Model:", model)

    # Train the model on the training data
    model.fit(x_train, y_train)

    # Make predictions on the testing data
    prediction = model.predict(x_test)

    # Evaluate model performance using various metrics

    # Accuracy: Proportion of correct predictions
    accuracy = accuracy_score(prediction, y_test)
    print("Accuracy Score:", accuracy)

    # Recall: Proportion of true positives correctly identified
    recall = recall_score(prediction, y_test)
    print("Recall Score:", recall)

    # F1 Score: Harmonic mean of precision and recall
    f1 = f1_score(prediction, y_test)
    print("F1 Score:", f1)

    # Create a heatmap of the confusion matrix
    sns.heatmap(confusion_matrix(prediction, y_test), annot=True, fmt='0.2f', linewidth=0.9)
    print('\n','\n')
```
    
#### Thank you for reading, and I hope this article was helpful 😊.

