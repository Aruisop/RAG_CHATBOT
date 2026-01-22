import os
import asyncio
import time
from typing import AsyncGenerator
from dotenv import load_dotenv

load_dotenv()

from groq import AsyncGroq 
from handlers.legal import LegalHandler
from database import VectorDB

class LegalAssistant:
    def __init__(self):
        # Initialize Async Client for non-blocking operations
        self.client = AsyncGroq()
        self.collection_name = "legal_docs_v1"
        self.file_path = os.path.join("data", "sampledata.pdf")
        
        # Initialize DB and Handler immediately
        print("Initializing High-Performance Legal Engine...")
        self.handler = LegalHandler()
        self.db = VectorDB(collection_name=self.collection_name)
        
        # State tracking
        self.is_ready = False

    async def initialize_knowledge_base(self):
        """
        Smart Ingestion: Only processes the PDF if the DB is empty.
        This saves massive time on restarts.
        """
        try:
            doc_count = self.db.count() 
        except:
            doc_count = 0 

        if doc_count > 0:
            print(f"Knowledge base loaded ({doc_count} chunks cached). Skipping ingestion.")
            self.is_ready = True
            return

        if not os.path.exists(self.file_path):
            print(f"Error: File '{self.file_path}' not found.")
            return

        print("New data detected. Ingesting & Chunking (this happens once)...")
        start_time = time.time()
        
        # Ingest
        raw_text = self.handler.ingest(self.file_path)
        chunks = await self.handler.chunk(raw_text)
        
        # Index
        self.db.add_documents(chunks)
        
        elapsed = time.time() - start_time
        print(f"Ingestion complete in {elapsed:.2f}s.")
        self.is_ready = True

    async def generate_response_stream(self, query: str) -> AsyncGenerator[str, None]:
        """
        Generates a streaming response. 
        This is the key to 'latency-free' perception.
        """
        if not self.is_ready:
            yield "System is still initializing..."
            return

        # 1. Fast Retrieval
        # improved accuracy: Fetch slightly more, then let the LLM filter
        context_docs = self.db.retrieve(query, top_k=7) 

        if not context_docs:
            yield "No relevant legal clauses found in the provided documents."
            return

        formatted_context = "\n\n---\n\n".join(context_docs)

        # 2. High-Accuracy Prompt
        system_prompt = """
        You are an expert Legal Contract Analyst.
        
        RULES:
        1. ANALYZE the provided context clauses first.
        2. ANSWER specific to the user's question.
        3. CITE exact Article numbers or Section headers (e.g., [Section 2.1]) for every claim.
        4. If the info is missing, state clearly: "The provided document does not mention..."
        5. Be concise but legally precise.
        """

        # 3. Stream from Groq (Async)+ low temp for legal docs
        try:
            stream = await self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{formatted_context}\n\nQuestion: {query}"}
                ],
                stream=True, 
                temperature=0.3, 
                max_tokens=1024
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"\n[Error generating response: {str(e)}]"

async def main():
    assistant = LegalAssistant()
    await assistant.initialize_knowledge_base()

    print("\nLEGAL AI READY. (Type 'exit' to quit)")
    print("--------------------------------------------------")

    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ["exit", "quit"]:
            break

        print("AI: ", end="", flush=True)
        
        # Simulate frontend consuming the stream
        async for token in assistant.generate_response_stream(user_query):
            print(token, end="", flush=True)
        print()

if __name__ == "__main__":
    asyncio.run(main())