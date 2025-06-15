import boto3
import os

class KnowledgeRetriever:
    def __init__(self):
        self.bedrock_agent_runtime = boto3.client('bedrock-agent-runtime')
        self.knowledge_base_id = "UJIRHUQ80C"

    def retrieve_documents(self, query: str) -> list[str]:
        """
        Retrieve relevant documents from the knowledge base using the provided query.
        Args:
            query (str): The query to search for in the knowledge base.
        Returns:
            list: A list of relevant documents or an error message if retrieval fails.
        """
        try:
            response = self.bedrock_agent_runtime.retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={'text': query}
            )
            print(response)
            documents = [
                        doc['content']['text'] 
                        for doc in response.get('retrievalResults', []) 
                        if 'text' in doc.get('content', {})
                    ]
            return documents or ["No relevant documents found."]
        except Exception as e:
            print(f"[Retriever Error] {e}")
            return ["[Error retrieving documents]"]

if __name__ == "__main__":
    # Example usage
    retriever = KnowledgeRetriever()
    
    query = "What is the process for applying for a loan on LendenClub?"
    documents = retriever.retrieve_documents(query)
    
    print("Retrieved Documents:")
    for doc in documents:
        print(f"- {doc}")