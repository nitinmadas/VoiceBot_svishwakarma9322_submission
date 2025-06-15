import json
import boto3
import os
from dotenv import load_dotenv
import yaml
from modules.utils import is_hindi, get_context

load_dotenv()

class LeeChatbot:
    def __init__(self):
        self.bedrock_agent_runtime = boto3.client('bedrock-agent-runtime')
        self.bedrock_runtime = boto3.client('bedrock-runtime')
        self.knowledge_base_id = os.getenv('knowledge_base_id')
        self.conversation_history = []

        with open('config/config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_id = self.config['model_id']
        self.max_tokens = self.config['max_tokens']

    def get_documents(self, query):
        try:
            response = self.bedrock_agent_runtime.retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={'text': query}
            )
            return [doc['content']['text'] for doc in response['retrievalResults']]
        except Exception as e:
            print(f"Error: {e}")
            return []

    def generate_response(self, question):
        in_hindi = is_hindi(question, self.config)
        documents = self.get_documents(question)
        context = "\n".join(documents)
        history = get_context(self.conversation_history)

        if in_hindi:
            prompt = f"""आप Lee हैं, Lenden Club के customer service rep। बहुत छोटे, friendly जवाब दें।

Context: {context}
{history}
User: {question}
बहुत संक्षिप्त जवाब दें (1-2 sentences max):"""
        else:
            prompt = f"""You are Lee from Lenden Club customer service. Give very short, friendly responses.

Context: {context}
{history}
User: {question}
Keep response very brief (1-2 sentences max):"""

        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                contentType='application/json',
                accept='application/json',
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.max_tokens,
                    "messages": [{"role": "user", "content": prompt}]
                })
            )
            result = json.loads(response['body'].read())
            answer = result['content'][0]['text'].strip()
            self.conversation_history.append((question, answer))
            self.conversation_history = self.conversation_history[-5:]
            return answer
        except Exception as e:
            print(f"Error: {e}")
            return "नमस्ते! मैं Lee हूं। तकनीकी समस्या है। कृपया अपने manager से संपर्क करें।" if in_hindi else "Hi! I'm Lee. Having technical issues. Please contact your relationship manager."

    def chat(self):
        print("💬 Hi! I'm Lee from Lending Club! How can I help? / नमस्ते! मैं Lee हूं!")
        print("Type 'quit' to exit.\n")
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye', 'धन्यवाद']:
                print("Lee: Thanks! Have a great day! / धन्यवाद! 😊")
                break
            if not user_input:
                print("Lee: What would you like to know?")
                continue
            print("Lee:", self.generate_response(user_input))

    def ask(self, question):
        return self.generate_response(question)

    def clear_history(self):
        self.conversation_history = []
        return "History cleared!"
