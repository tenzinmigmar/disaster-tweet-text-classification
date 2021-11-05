# --- Importing dependencies
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import unidecode

train_data = pd.read_csv('data/train (1).csv')
test_data = pd.read_csv('data/test (1).csv')

# --- Preprocessing class
class Preprocessing():

    tweet_id = 0

    def __init__(self, path):
        self.path = path
        self.input = pd.read_csv(self.path)
        self.full_data = self.return_data()
        self.id = Preprocessing.tweet_id
        self.tweet = self.remove_accents()[self.tweet_id]
        self.tokens = self.lemmatize()

        Preprocessing.tweet_id += 1

    def __len__(self):
        return len(self.tokens)

    def return_data(self):
        return [tweet.lower() for tweet in self.input['text']]

    def remove_accents(self):
        self.return_data()
        return [unidecode.unidecode(tweet) for tweet in self.return_data()]

    def tokenize(self):
        return word_tokenize(self.tweet)

    def remove_stopwords(self):
        self.tokenize()
        stop_words = set(stopwords.words('english'))
        return [token for token in self.tokenize() if token not in stop_words]

    def remove_special_chars(self):
        self.remove_stopwords()
        return [token for token in self.remove_stopwords() if token.isalnum()]

    def stemming(self):
        self.remove_special_chars()
        stemmer = PorterStemmer()
        return [stemmer.stem(token) for token in self.remove_special_chars()]

    def lemmatize(self):
        self.stemming()
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in self.stemming()]

### -- Creating objects
corpus = [Preprocessing('data/train (1).csv').tokens for i in range(train_data.shape[0])]
print(corpus)