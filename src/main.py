from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load training data
train_data = pd.read_csv('data/train_data.csv')
val_data = pd.read_csv('data/val_data.csv')
test_data = pd.read_csv('data/test_data.csv')

# Initialize CountVectorizer for Bag-of-Words
'''
Initializes the Bag-of-Words model.
max_features=5000: Limits the vocabulary to the 5,000 most frequent words across the dataset, reducing dimensionality.
stop_words='english': Automatically removes common English stop words (e.g., "the," "and," "is") to focus on meaningful words.
'''
bow_vectorizer = CountVectorizer(max_features=5000, stop_words='english')



# Fit on training data and transform
'''
bow_vectorizer.fit_transform(train_data['text']):

Fit: Learns the vocabulary from the training data.
Transform: Converts the text data (train_data['text']) into a sparse matrix of word counts.
X_train: Numerical representation of the training text data.
y_train = train_data['sentiment']:

Extracts the target labels (sentiments) from the training dataset to form y_train.
'''
X_train = bow_vectorizer.fit_transform(train_data['text'])
y_train = train_data['sentiment']

# Transform validation and test data
X_val = bow_vectorizer.transform(val_data['text'])
y_val = val_data['sentiment']

X_test = bow_vectorizer.transform(test_data['text'])
y_test = test_data['sentiment']

# Save the feature names (optional, for understanding BoW output)
feature_names = bow_vectorizer.get_feature_names_out()



# Logistic model #
batch_size = 10
epochs = 50
threshold = 0.5

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def training_logistic(x, y,learning_rate=0.01):
    n_samples, n_features = x.shape
    w = np.zeros(n_features)
    b = 0
    print(n_samples)

    for epoch in range(epochs):
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            X_batch = x[indices[start:end]]
            y_batch = y[indices[start:end]]

            z = X_batch @ w + b
            y_pred = sigmoid(z)

            dw = (1 / batch_size) * (X_batch.T @ (y_pred - y_batch)) 
            db = (1 / batch_size) * np.sum(y_pred - y_batch)

            w -= learning_rate * dw
            b -= learning_rate * db

    return w, b



def predict_logistic(X,w,b):
    z = X @ w + b
    y = sigmoid(z)
    return (y >= threshold).astype(int)

# Naive Bayes

def train_naive_bayes(X,y):
    n_samples, n_features = X.shape
    classes = np.unique(y)
    vocabulary = range(n_features)

    logprior = {}
    loglikelihood = {}

    for cls in classes:
        X_cls = X[y == cls]
        N_c = X_cls.shape[0]
        logprior[cls] = np.log(N_c / n_samples)

        word_count_in_class = np.sum(X_cls, axis=0)
        total_word_count = np.sum(word_count_in_class)

        loglikelihood[cls] =  np.log((word_count_in_class + 1) / (total_word_count + vocabulary))

    return logprior,loglikelihood,vocabulary,classes

def predict_naive_bayes(X,logprior,loglikelihood,classes):
    n_samples, n_features = X.shape
    predictions = []

    for i in range(n_samples):
        sum_posteriors = {}

        for cls in classes:
            sum_posteriors[cls] = logprior[cls]

            for j in range(n_features):
                if X[i,j]>0:
                    sum_posteriors[cls] += X[i,j] * loglikelihood[cls][j]

        predictions.append(max(sum_posteriors, key=sum_posteriors.get))

    return np.array(predictions)



def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

def main():
    # Step 1: Prepare the Data
    print("Preparing data...")
    # Convert the sparse matrix to dense and extract labels
    X_train_dense = X_train.toarray()
    X_val_dense = X_val.toarray()
    X_test_dense = X_test.toarray()

    y_train_np = y_train.to_numpy()
    y_val_np = y_val.to_numpy()
    y_test_np = y_test.to_numpy()

    # Train LOGISTIC ---------------------------------------------------
    # print("Training logistic regression model...\n")
    # w, b = training_logistic(X_train_dense, y_train_np,learning_rate=0.01)

    # #  Make Predictions
    # print("Making predictions...\n")
    # y_val_pred = predict_logistic(X_val_dense, w, b)
    # y_test_pred = predict_logistic(X_test_dense, w, b)

    # # Step 4: Evaluate the Model
    # print("Evaluating model on validation set...\n")
    # evaluate(y_val_np, y_val_pred)

    # print("\nEvaluating model on test set...\n")
    # evaluate(y_test_np, y_test_pred)

    # NAIVE BAYES ---------------------------------------------------
    print("Training Naive Bayes model...\n")
    logprior, loglikelihood, vocabulary, classes = train_naive_bayes(X_train_dense, y_train_np)

    print("Making predictions...\n")
  
    y_val_pred = predict_naive_bayes(X_val_dense, logprior, loglikelihood, classes)
   
    y_test_pred = predict_naive_bayes(X_test_dense, logprior, loglikelihood, classes)

    print("Evaluating model on validation set...\n")
    evaluate(y_val_np, y_val_pred)

    print("\nEvaluating model on test set...\n")
    evaluate(y_test_np, y_test_pred)



if __name__ == "__main__":
    main()






