## Natural Language Processing - predict if a new restaurant review is postive or negative

## Importing the libriaries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importint the dataset (add paramiters for tab in tsv file and ignoring quoting with 3)
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) 

##### CLEANIG TEXT:
import re
##                            
review = re.sub('[^a-zA-Z]',                                  ## not the letters a-z/A-Z..
                ' '              ,                            ## space to separate characters    
                dataset['Review'][0]     )                    ## in a first review....
                
                
## put all the letters in a lowercase:
review = review.lower()


## get rid of irrelevant and stemming words:
import nltk   
nltk.download('stopwords')          ## stopwords list
from nltk.corpus import stopwords
review = review.split()          ## sentence to the list

## getting the root of every word (stemming):
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]  ##keep the words that are not in a stopwords
        ##stemming                                     ## set makes algorithm faster


review = ' '.join(review)               ## join he words with a space
                                           
#### INITIALIZE FOR LOOP:

corpus = []
for i in range(0,1000):
        review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])    ## all the indexes
        review = review.lower()
        review = review.split() 
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)

#### Creating the Bag of Words model - all the unique words 0-1 table for 1000 reviews (tokenization process)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)     ## get rid of non-relevant words
X = cv.fit_transform(corpus).toarray()
                        ### to create matrix of features
y = dataset.iloc[:,1].values   ## original outcome of the reviews


############################### Train the prediction model with Naive Bayes ##################################################    
         
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
## Confusion Matrix:
##   67	50
##   20	113
## Accuracy 72%


############################### Train the prediction model with Decision Tree ##################################################

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

## Feature Scaling - Not applicable as we don't deal with euclidean distance
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)   ## Entropy to measuer better quality of a dec. trees
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


## Confusion Matrix:
##   91	26
##   43 90
## Accuracy 72%



############################### Train the prediction model with Random Forest ##################################################


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Fitting Random Forest Classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 500,   ## number of trees - prevent overfitting
                                    criterion = 'entropy',  ## quality of a split measure according to information entropy theory
                                    random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

## Confusion Matrix:
##   105	12
##   65	  68
## Accuracy 69%

############################### Train the prediction model with Kernel SVM ##################################################



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0 )
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

## Confusion Matrix:
##   79	38
##   28	105
## Accuracy 73%


############################### Train the prediction model with SVM Classification ##################################################


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0) ## CHoose different kernel for better prediction
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

## Confusion Matrix:
##   94	23
##   49	84
## Accuracy 71%

############################### Train the prediction model with K-Nearest Neighbours ##################################################

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
## metric - minkowski & p = 2 for Euclidean distance instead Manhattan
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

## Confusion Matrix:
##   88	29
##   67	66
## Accuracy 61%

############################### Train the prediction model with Logistic Regression ##################################################


# Splitting the dataset into the Training set and Test set 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling (mean by deafult) - necessary as age compare to salary values range is to wide
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

## Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression      ## Class starts with capital letters, while function has underscore
classifier = LogisticRegression(random_state = 0)        ## create an object for a model
classifier.fit(X_train, y_train)                         ## fit an object into training set       

## Predicting the Test set results
y_pred = classifier.predict(X_test)

## Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix     ## Class starts with capital letters, while function has underscore
cm = confusion_matrix(y_test, y_pred)           ## Object that compares prediction to real values on a test set

## Confusion Matrix:
##   89	28
##   39	94
## Accuracy 73%



















                                            