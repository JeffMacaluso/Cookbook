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
df = df.assign(tokenized_corpus=df['corpus'].map(tokenizer.tokenize))  # If the corpus is a column in a dataframe

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

## Topic modeling
def topic_model_lda(processed_corpus, num_topics=5, num_words=4):
    '''
    Uses Latent Dirichlect Allocation for topic modeling
    Borrowed from https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21
    
    Inputs: 
        - Processed_corpus: The corpus of text that has already been tokenized/stemmed/etc.
        - num_topics: The number of topics to discover in the text
        - num_words: The number of words per topic to print in the output
    
    Output: A pretty printed list of words in each topic and the probability associated with htem
    
    TODO: Print interpretation of the numbers
    '''
    import gensim
    
    # Additional processing before modeling
    dictionary = corpora.Dictionary(processed_corpus)
    corpus = [dictionary.doc2bow(text) for text in processed_corpus]
    
    # Performing LDA
    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    topics = lda_model.print_topics(num_words=num_words)
    
    # Reporting the topics
    for topic in topics:
        print('Topic {0}:'.format(topic[0]))  # Printing the topic number
        topic_words = topic[-1].replace('"', '').replace(' ', '')  # Removing white space and quotes
        topic_words = [word.split('*') for word in topic_words.split('+')]  # Splitting into one item for the word and one for the probability
        [print(' {0}: {1}'.format(x[1], x[0])) for x in topic_words]  # Printing the results as word: probability
        print()  # New line
