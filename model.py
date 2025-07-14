import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

def train_model():
    df = pd.read_csv('data/career_data.csv')

    # Encode the 'interest' column directly in the original DataFrame
    le = LabelEncoder()
    df['interest'] = le.fit_transform(df['interest'])

    # Now extract features and target safely
    X = df[['math', 'science', 'english', 'interest']]
    y = df['career']

    # Train KNN
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)

    return model, le
