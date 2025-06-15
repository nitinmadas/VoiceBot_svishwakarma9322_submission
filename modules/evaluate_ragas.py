from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    faithfulness,
    answer_relevancy,
    context_recall,
)
import boto3
from langchain_aws import ChatBedrock # <-- IMPORT THE CORRECT CLASS

bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")  # Change to your preferred r

llm = ChatBedrock(
    # Ensure your AWS credentials are configured (e.g., via environment variables)
    region_name="us-west-2",  # Change to your preferred region if needed
    model_id="anthropic.claude-3-sonnet-20240229-v1:0", # <-- CORRECT MODEL ID
    model_kwargs={"temperature": 0.0} # Pass model parameters here
)

import json
from retriever import KnowledgeRetriever
lee = KnowledgeRetriever()

# Load QA data
with open("qa2_dataset.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)


questions, answers, ground_truths, contexts = [], [], [], []

from response_gen import agent
counter = 0
config = {"configurable": {"thread_id": "lendencol-thread-3"}}

for item in qa_data:
    question = item["question"]
    gt = item["answer"]
    ctx = lee.retrieve_documents(question)
    response = agent.invoke(    
             {"messages": [{"role": "user", "content": str(question)}]},
             config=config
                )  # Ensure it's string

    ans = response["messages"][-1].content
    

    questions.append(question)
    ground_truths.append(gt)
    contexts.append(ctx)
    answers.append(ans)

    print(f"{counter} Question: {question}\nAnswer: {ans}")
    counter += 1
# Build dataset
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "reference": ground_truths,
}
# Optionally export the data to a JSON file for inspection
with open("ragas_eval_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
dataset = Dataset.from_dict(data)



result = evaluate(
    dataset=dataset,
    metrics=[context_precision, faithfulness, answer_relevancy, context_recall],
    llm=llm
)


print(result.to_pandas().T)
# Save the evaluation results to a CSV file
result.to_pandas().to_csv("ragas_evaluation_results.csv", index=False)
