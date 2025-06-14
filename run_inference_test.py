import boto3
import json
from dotenv import load_dotenv
load_dotenv()  # looks for a file named `.env` in the current directory
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime')

response = bedrock_agent_runtime.retrieve(
    knowledgeBaseId='USH0TCG5JK',
    retrievalQuery={'text': 'What is the lengen policy?'}
)

documents = [doc['content']['text'] for doc in response['retrievalResults']]

import json
import boto3

bedrock_runtime = boto3.client('bedrock-runtime')

context = "\n\n".join(documents)

# ✅ Claude-specific prompt formatting
prompt = f"""
\n\nHuman:
You are a helpful assistant. Use the following context to answer the user's question.

Context:
{context}

Question: What is the return policy?

\n\nAssistant:
"""

response = bedrock_runtime.invoke_model(
    modelId='anthropic.claude-v2',  # You can switch to v3 if needed
    contentType='application/json',
    accept='application/json',
    body=json.dumps({
        "prompt": prompt,
        "max_tokens_to_sample": 300
    })
)

generated_text = json.loads(response['body'].read())['completion']
print(generated_text.strip())

