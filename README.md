
# Handling Different Data Types  
## Topic: Image, Tabular, and Text Data with ANNs  

##  Summary

In this lesson, we will:
- Learn how to **preprocess different types of data** (images, tabular, and text) for use with **Artificial Neural Networks (ANNs)**.
- Understand the steps to handle **image data** using flattening and normalization.
- Handle **tabular data** (numerical + categorical) using encoding and scaling.
- Get introduced to **text preprocessing** with tokenization and embedding.
- Build **practical ANN models** for both **tabular** and **image** data types.

---

## 1.  Why Data Type Matters in Neural Networks

Different data types come in **different shapes and meanings**. Before feeding them into a neural network, we need to **make them digestible**—like preparing food before serving it.

---

###  Real-world Analogy

Think of a neural network like a **multilingual receptionist**:
- You can feed it **pictures (images)**, **numbers (tabular)**, or **sentences (text)**.
- But before it understands, you must **translate everything into a common language**: **numerical vectors**.

---

## 2.  Handling Image Data

Neural networks don’t inherently "see" images. They only see numbers!

###  Steps:
- **Flatten** the image: convert 2D pixels into 1D array.
- **Normalize**: scale pixel values to be between 0 and 1 (helps in faster and stable training).

---

###  Example – Preprocessing and Classifying Digits (MNIST)

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build model
model = Sequential([
    Flatten(input_shape=(28, 28)),   # Flatten image to 784 vector
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 digits
])

# Compile & Train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Image Classification Accuracy: {acc:.4f}")
````

---

###  Real-World Applications

* Classifying handwritten forms
* Detecting digits on bank cheques
* Basic vision tasks without CNNs (for small images)

---

## 3.  Handling Tabular Data (Categorical + Numerical)

Tabular data is **structured like a spreadsheet**.

> Neural networks need **numerical vectors**, so we convert all inputs to numbers.

---

###  Key Preprocessing Steps

#### A. For **Numerical** Features:

* **Standardize** or **normalize** them (e.g., using `StandardScaler` or `MinMaxScaler`).

#### B. For **Categorical** Features:

* **Encode** them (e.g., One-Hot Encoding or Embedding)

---

###  Example – Predicting Survival on Titanic Dataset

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_openml

# Load Titanic dataset
df = fetch_openml('titanic', version=1, as_frame=True)['frame']
df = df[['pclass', 'sex', 'age', 'fare', 'embarked', 'survived']].dropna()

# Features and target
X = df.drop('survived', axis=1)
y = df['survived'].astype('int')

# Preprocessing
numeric = ['age', 'fare']
categorical = ['pclass', 'sex', 'embarked']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric),
    ('cat', OneHotEncoder(), categorical)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Build model
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile & Train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Tabular Model Accuracy: {acc:.4f}")
```

---

###  Real-World Applications

* Predicting loan defaults
* Insurance claim predictions
* Employee attrition modeling

---

## 4.  Handling Text Data

Text is **unstructured** and needs special treatment.

---

###  Key Preprocessing Steps

1. **Tokenization**: Break sentences into words or tokens.
2. **Vectorization**: Convert tokens into numbers.
3. **Embedding**: Learn dense numerical representations of words.

---

### Analogy

If words are **books**, embedding is like putting them into a **library system** where similar books are stored on nearby shelves.

---

### This topic is covered in-depth in the NLP section. Here’s a quick teaser:

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

texts = ["I love AI", "AI is the future", "I love future"]
labels = [1, 1, 0]

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
X_seq = tokenizer.texts_to_sequences(texts)
X_pad = pad_sequences(X_seq, maxlen=5)

print("Tokenized and padded sequences:")
print(X_pad)
```

---

##  Summary Comparison Table

| Data Type | Preprocessing Required | ANN-ready Shape         | Real-world Uses                          |
| --------- | ---------------------- | ----------------------- | ---------------------------------------- |
| Image     | Flatten + Normalize    | 1D vector               | Digit/image classification               |
| Tabular   | Encoding + Scaling     | All numeric features    | Finance, healthcare, HR                  |
| Text      | Tokenize + Pad + Embed | Sequences or embeddings | Sentiment, chatbots, text classification |

---

##  Final Thoughts

* **Images**, **tables**, and **text** are all usable in ANNs, but each needs a different **preparation process**.
* Always **convert your data into numerical vectors** and normalize/standardize them.
* The **Sequential API** is powerful enough to build models for these datasets with ease.

---


