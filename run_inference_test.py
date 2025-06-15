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



import boto3
import json
from dotenv import load_dotenv
import re
import os

load_dotenv()

class LeeChatbot:
    def __init__(self):
        self.bedrock_agent_runtime = boto3.client('bedrock-agent-runtime')
        self.bedrock_runtime = boto3.client('bedrock-runtime')
        self.knowledge_base_id = os.getenv('knowledge_base_id')
        self.conversation_history = []
        self.name = "Lee"
    
    def is_hindi(self, text):
        """Simple Hindi detection"""
        hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
        hindi_words = ['kya', 'hai', 'main', 'mera', 'aap', 'hum', 'kaise', 'kahan']
        hindi_word_count = sum(1 for word in hindi_words if word in text.lower())
        
        return (hindi_chars > len(text) * 0.2) or (hindi_word_count >= 2)
    
    def get_documents(self, query):
        """Get relevant documents from knowledge base"""
        try:
            response = self.bedrock_agent_runtime.retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={'text': query}
            )
            return [doc['content']['text'] for doc in response['retrievalResults']]
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def get_context(self):
        """Get recent conversation context"""
        if not self.conversation_history:
            return ""
        
        context = "\nRecent chat:\n"
        for user_msg, bot_msg in self.conversation_history[-2:]:  # Last 2 exchanges
            context += f"User: {user_msg}\nLee: {bot_msg}\n"
        return context
    
    def generate_response(self, question):
        """Generate short, conversational response"""
        is_hindi = self.is_hindi(question)
        documents = self.get_documents(question)
        context = "\n".join(documents)
        chat_history = self.get_context()
        
        # Shorter, more focused prompts
        if is_hindi:
            prompt = f"""आप Lee हैं, Lending Club के customer service rep। बहुत छोटे, friendly जवाब दें।

                Context: {context}
                {chat_history}

                User: {question}

                बहुत संक्षिप्त जवाब दें (1-2 sentences max):"""
        else:
            prompt = f"""You are Lee from Lending Club customer service. Give very short, friendly responses.

                    Context: {context}
                    {chat_history}

                    User: {question}

                    Keep response very brief (1-2 sentences max):"""

        try:
            response = self.bedrock_runtime.invoke_model(
                modelId='anthropic.claude-3-sonnet-20240229-v1:0',
                contentType='application/json',
                accept='application/json',
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 150,  # Reduced for shorter responses
                    "messages": [{"role": "user", "content": prompt}]
                })
            )
            
            result = json.loads(response['body'].read())
            answer = result['content'][0]['text'].strip()
            
            # Store conversation
            self.conversation_history.append((question, answer))
            
            # Keep only last 5 exchanges
            if len(self.conversation_history) > 5:
                self.conversation_history = self.conversation_history[-5:]
            
            return answer
            
        except Exception as e:
            print(f"Error: {e}")
            return "Hi! I'm Lee. Having technical issues. Please contact your relationship manager." if not is_hindi else "नमस्ते! मैं Lee हूं। तकनीकी समस्या है। कृपया अपने manager से संपर्क करें।"
    
    def chat(self):
        """Interactive chat session"""
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
            
            response = self.generate_response(user_input)
            print(f"Lee: {response}\n")
    
    def ask(self, question):
        """Single question method"""
        return self.generate_response(question)
    
    def clear_history(self):
        """Clear chat history"""
        self.conversation_history = []
        return "History cleared!"

# Simple usage functions
def chat_with_lee(question):
    """Quick question to Lee"""
    lee = LeeChatbot()
    return lee.ask(question)

def start_chat():
    """Start interactive chat"""
    lee = LeeChatbot()
    lee.chat()

# Run chatbot
if __name__ == "__main__":
    # start_chat()
    
    # chat_with_lee("What is the lengen")  # Example usage
