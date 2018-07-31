import os
import sys
import time
import re
import nltk

print(time.strftime('%Y/%m/%d %H:%M'))
print('OS: ', sys.platform)
print('CPU Cores:', os.cpu_count())
print('Python: ', sys.version)
print('NLTK: ', nltk.__version__)

### Tokenizing
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')  # By blank space
corpus = corpus.apply(tokenizer.tokenize)  # If in a dataframe


### Removing stop words
filtered_words = []
for document in corpus:
    filtered_words.append([
        word.lower() for word in document
        if word.lower() not in nltk.corpus.stopwords.words('english')
    ])


### Lemmatizing
# Setting the Lemmatization object
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

# Looping through the words and appending the lemmatized version to a list
stemmed_words = []
for row in df['tokens']:
    stemmed_words.append([
        # Verbs
        lemmatizer.lemmatize(  
            # Adjectives
            lemmatizer.lemmatize(  
                # Nouns
                lemmatizer.lemmatize(word.lower()), 'a'), 'v')
        for word in row
        if word.lower() not in nltk.corpus.stopwords.words('english')])


### TF-IDF
# Creating the sklearn object
from sklearn.feature_extraction import text as sktext
tfidf = sktext.TfidfVectorizer(smooth_idf=False)

# Transforming our 'tokens' column into a TF-IDF matrix and then a data frame
tfidf_df = pd.DataFrame(tfidf.fit_transform(corpus).toarray(), 
                        columns=tfidf.get_feature_names())

# Removing sparse columns
tfidf_df = tfidf_df[tfidf_df.columns[tfidf_df.sum() > 2.5]]

# Removing digits
tfidf_df = tfidf_df.filter(regex=r'^((?!\d).)*$')
