import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Đọc dữ liệu
df = pd.read_csv('data.csv')

# Tạo vector TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['label'].apply(lambda x: 1 if x == 'sensitive' else 0)

# Train Logistic Regression
model = LogisticRegression()
model.fit(X, y)

# Lưu model và vectorizer
joblib.dump(model, 'model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

print("Training hoàn tất, model và vectorizer đã được lưu.")
