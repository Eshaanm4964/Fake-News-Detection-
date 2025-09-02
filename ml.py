#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re 
import nltk 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,r2_score,classification_report,confusion_matrix

#Importing and learning about data:

data = pd.read_csv(r"C:\Users\eshaa\OneDrive\Desktop\Fake News Detection\fake_and_real_news.csv")
# %%
data.head(10)
# %%
data.info()
# %%
data.describe()
# %%
data.mode()
# %%
data.tail(10)
# %%
data.columns

# %%
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
# %%
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# %%
data["clean_text"] = data["Text"].apply(preprocess)

# %%
data.head(10)
# %%
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df = 0.7)
X = tfidf.fit_transform(data["clean_text"])
# %%

le = LabelEncoder()
y = le.fit_transform(data["label"])

# %%
data.head(10)
# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
data.head(10)
# %%
print(X.shape)

# %%
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# %%
