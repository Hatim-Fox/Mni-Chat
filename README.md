# Mni-Chat

A simple and intelligent chatbot that uses TF-IDF vectorization and cosine similarity to provide accurate answers based on a pre-trained dataset.

## Features

- **TF-IDF Vectorization**: Converts text questions into numerical representations
- **Cosine Similarity**: Finds the most similar question in the dataset
- **Threshold-based Filtering**: Avoids returning incorrect answers below a similarity threshold (0.2)
- **Interactive CLI**: Easy-to-use command-line interface

## Requirements

- Python 3.7+
- scikit-learn
- pandas

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Hatim-Fox/Mni-Chat.git
cd Mni-Chat
```

2. Install required dependencies:
```bash
pip install scikit-learn pandas
```

## Usage

1. Prepare your dataset in `dataset.csv` with two columns: `question` and `answer`

2. Run the chatbot:
```bash
python Mni_Chat.py
```

3. Start chatting:
```
Hello! Ask me anything you want (type exit to end the chat)
Me: Your question here
Bot: Answer based on the dataset
```

4. Type `exit` or `bye` to end the conversation

## Dataset Format

The `dataset.csv` file should have the following structure:

| question | answer |
|----------|--------|
| How are you? | I'm doing great, thank you! |
| What's your name? | I'm Mni-Chat, your assistant |

## How It Works

1. The chatbot loads questions and answers from `dataset.csv`
2. All questions are converted to TF-IDF vectors
3. When a user asks a question, it's converted to a TF-IDF vector
4. Cosine similarity is calculated between the user's input and all dataset questions
5. The answer with the highest similarity score is returned (if score ≥ 0.2)
6. If no match is found, the chatbot prompts the user to rephrase their question

## License

This project is open source and available under the MIT License.
