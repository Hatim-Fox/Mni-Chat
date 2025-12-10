from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load the data
df = pd.read_csv("dataset.csv")

questions = df["question"].tolist()
answers = df["answer"].tolist()

# Convert questions to TF-IDF numeric representation
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

def chat_bot(user_input):
    # Convert the new question to TF-IDF representation
    user_vec = vectorizer.transform([user_input])

    # Calculate similarity with all questions in the dataset
    similarities = cosine_similarity(user_vec, X)

    # Find the highest similarity score
    idx = similarities.argmax()
    score = similarities[0][idx]

    # Minimum threshold to avoid returning a wrong answer
    if score < 0.2:
        return "I didn't understand your question, try rephrasing it differently."

    return answers[idx]

# Test the chatbot
print("Hello! Ask me anything you want (type exit to end the chat)")

while True:
    user_input = input("Me: ")

    if user_input.lower() in ["bye", "exit"]:
        print("Goodbye!")
        break

    response = chat_bot(user_input)
    print("Bot:", response)
