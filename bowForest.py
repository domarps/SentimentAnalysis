import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup             
from sklearn.feature_extraction.text import CountVectorizer
#nltk.download()


stopWords = stopwords.words("english") 




def reviewToWords(raw_review):
	if(pd.isnull(raw_review)):
		return []
	review_text = raw_review
	letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
	words = letters_only.lower().split()
	stops = set(stopwords.words("english")) 
	meaningful_words = [w for w in words if not w in stops]
	return( " ".join( meaningful_words ))


data = pd.read_csv('product_recommendation.csv')
print data.shape
print data.columns.values

# print data["pr_review_content"][0]
# example1 = BeautifulSoup(data["pr_review_content"][0])  
# print example1.get_text()

# Get the number of reviews based on the dataframe column size
num_reviews = data["pr_review_content"].size

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

print "Cleaning and parsing the training set movie reviews...\n"
clean_train_reviews = []
for i in xrange( 0, num_reviews ):
    # If the index is evenly divisible by 1000, print a message
    if( (i+1)%1000 == 0 ):
        print "Review %d of %d\n" % ( i+1, num_reviews )                                                                    
    clean_train_reviews.append(reviewToWords(data["pr_review_content"][i]))

print "Creating the bag of words...\n"
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

#[item.lower() for item in clean_train_reviews]
print len(clean_train_reviews)

for review in clean_train_reviews:
	print review
	train_data_features = vectorizer.fit_transform(clean_train_reviews)
#train_data_features = train_data_features.toarray()

# #print train_data_features.shape()
