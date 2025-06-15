# import json
# import pandas as pd
# from ragas import evaluate
# from ragas.metrics import answer_relevancy, faithfulness, context_precision
# from datasets import Dataset
# from modules.response_gen import LeeChatbot

# def load_qa_data(filepath):
#     with open(filepath, 'r') as f:
#         return json.load(f)

# def evaluate_ragas(file_path="qa2_dataset.json"):
#     qa_pairs = load_qa_data(file_path)

#     lee = LeeChatbot()
#     questions = []
#     answers = []
#     generated_answers = []
#     contexts = []

#     for qa in qa_pairs:
#         question = qa["question"]
#         expected = qa["answer"]
#         response = lee.ask(question)
#         docs = lee.get_documents(question)

#         questions.append(question)
#         answers.append(expected)
#         generated_answers.append(response)
#         contexts.append("\n".join(docs))

#     df = pd.DataFrame({
#         "question": questions,
#         "answer": generated_answers,
#         "contexts": contexts,
#         "ground_truth": answers
#     })

#     # Convert to HuggingFace dataset
#     dataset = Dataset.from_pandas(df)

#     # Evaluate with ragas
#     results = evaluate(
#         dataset=dataset,
#         metrics=[
#             answer_relevancy,
#             faithfulness,
#             context_precision
#         ]
#     )

#     print("📊 RAGAS Evaluation Results:\n")
#     print(results)

# if __name__ == "__main__":
#     evaluate_ragas()
'''
from datasets import Dataset
import json
import boto3
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas import evaluate

# Load QA JSON (your `qa2_dataset.json`)
with open("qa2_dataset.json", "r") as f:
    qa_data = json.load(f)

# Convert to HuggingFace Dataset format
dataset = Dataset.from_list([
    {
        "question": item["question"],
        "answer": item["answer"],
        "contexts": [],  # Add retrieved context here if available
        "ground_truth": item["answer"]
    }
    for item in qa_data
])

# AWS Bedrock LLM using boto3 client
bedrock_client = boto3.client("bedrock-runtime")

# Run RAGAS Evaluation
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
)

# Print Results
print(result)
'''

import json
from response_gen import LeeChatbot
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load your model for similarity (semantic embedding model)
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose another

# Load ground truth dataset
with open('data/q2_dataset.json', 'r') as f:
    gt_data = json.load(f)

# Initialize agent
chatbot = LeeChatbot()

# Evaluation storage
results = []

for item in tqdm(gt_data, desc="Evaluating"):
    question = item['question']
    true_answer = item['answer']

    # Clear chat history before each run
    chatbot.clear_history()

    # Generate agent answer
    generated_answer = chatbot.ask(question)

    # Calculate semantic similarity
    embeddings = model.encode([true_answer, generated_answer])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    results.append({
        'question': question,
        'ground_truth': true_answer,
        'generated': generated_answer,
        'similarity_score': score
    })

# Save results
with open('evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print average score
avg_score = sum([r['similarity_score'] for r in results]) / len(results)
print(f"Average Semantic Similarity Score: {avg_score:.4f}")
