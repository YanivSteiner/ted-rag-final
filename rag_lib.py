import os
import pandas as pd
import time
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

class TedRagSystem:
    def __init__(self, csv_path: str, chunk_size: int = 1000, overlap_ratio: float = 0.1, top_k: int = 5):
        # FIX 1: Connect to the Course Server with the correct Base URL
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.llmod.ai/v1" 
        )
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = "ted-rag-index"
        self.csv_path = csv_path
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.top_k = top_k
        
        self._initialize_index()
        self.index = self.pc.Index(self.index_name)

    def _initialize_index(self):
        existing_indexes = [i.name for i in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            print(f"Creating new Pinecone index: {self.index_name}...")
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
            print("Index created successfully!")
        else:
            print(f"Index {self.index_name} already exists.")

    def load_and_process_data(self, limit: int = 50):
        stats = self.index.describe_index_stats()
        if stats.total_vector_count > 0:
            print(f"Index already contains {stats.total_vector_count} vectors. Skipping processing.")
            return

        print("Processing CSV and uploading to Pinecone...")
        df = pd.read_csv(self.csv_path)
        
        if limit:
            df = df.head(limit)

        vectors_to_upsert = []
        
        for _, row in df.iterrows():
            if pd.isna(row['transcript']): continue

            base_info = f"Title: {row['title']}\nSpeaker: {row['speaker_1']}\n"
            text_len = len(row['transcript'])
            step = int(self.chunk_size * (1 - self.overlap_ratio))
            
            for i in range(0, text_len, step):
                chunk_text = row['transcript'][i : i + self.chunk_size]
                full_text = base_info + "Transcript: " + chunk_text
                chunk_id = f"{row['talk_id']}_{i}"
                
                # FIX 2: Using the EXACT Embedding Model Name from your screenshot
                res = self.client.embeddings.create(
                    input=full_text, 
                    model="RPRTHPB-text-embedding-3-small"
                )
                embedding = res.data[0].embedding
                
                metadata = {
                    "text": full_text,
                    "original_text": chunk_text,
                    "title": row['title'],
                    "talk_id": str(row['talk_id'])
                }
                
                vectors_to_upsert.append((chunk_id, embedding, metadata))
                
                if len(vectors_to_upsert) >= 100:
                    self.index.upsert(vectors=vectors_to_upsert)
                    vectors_to_upsert = []
        
        if vectors_to_upsert:
            self.index.upsert(vectors=vectors_to_upsert)
            
        print("Data upload to Pinecone complete.")

    def answer_query(self, query: str) -> Dict:
        # FIX 2 (Again): Using the EXACT Embedding Model Name
        query_res = self.client.embeddings.create(
            input=query, 
            model="RPRTHPB-text-embedding-3-small"
        )
        query_vec = query_res.data[0].embedding
        
        search_res = self.index.query(vector=query_vec, top_k=self.top_k, include_metadata=True)
        
        relevant_chunks = []
        context_texts = []
        
        for match in search_res['matches']:
            relevant_chunks.append({
                "talk_id": match['metadata']['talk_id'],
                "title": match['metadata']['title'],
                "chunk": match['metadata']['original_text'],
                "score": match['score']
            })
            context_texts.append(match['metadata']['text'])
            
        system_prompt = """You are a TED Talk assistant that answers questions strictly and only based on the TED dataset context provided to you (metadata and transcript passages).
                           You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context.
                           If the answer cannot be determined from the provided context, respond: "I don't know based on the provided TED data."
                           Always explain your answer using the given context, quoting or paraphrasing the relevant transcript or metadata when helpful."""
        context_str = "\n---\n".join(context_texts)
        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
        
        # FIX 3: Using the EXACT Chat Model Name from your screenshot
        response = self.client.chat.completions.create(
            model="RPRTHPB-gpt-5-mini", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        return {
            "response": response.choices[0].message.content,
            "context": relevant_chunks,
            "Augmented_prompt": {"System": system_prompt, "User": user_prompt}
        }