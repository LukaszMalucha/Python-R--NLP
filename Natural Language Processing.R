# Natural Language Processing

# Importing the dataset - tab as a separator, specify categorical variable as factors = false
dataset_original = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)

# Cleaning the texts
install.packages('tm')
install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))  ## specify source for corpus
corpus = tm_map(corpus, content_transformer(tolower))    ## lower cases
corpus = tm_map(corpus, removeNumbers)                   ## remove digits
corpus = tm_map(corpus, removePunctuation)               ## remove punctuation
corpus = tm_map(corpus, removeWords, stopwords())        ## remove irrelavant words (snowball)
corpus = tm_map(corpus, stemDocument)                    ## stemming words
corpus = tm_map(corpus, stripWhitespace)                 ## remove whitespace

# Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)                      ## keep 99% most frequest words
dataset = as.data.frame(as.matrix(dtm))                  ## change data frame into matrix
dataset$Liked = dataset_original$Liked                   ## add 'liked' column as dependent variable


# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)