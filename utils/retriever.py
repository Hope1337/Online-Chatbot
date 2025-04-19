import os
import pickle
import PyPDF2
import torch
import faiss
import numpy as np
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    # (Giữ nguyên hàm chunk_text của bạn)
    words = text.split()
    if len(words) < chunk_size:
        return [text.strip()]
    
    chunks = []
    start_idx = 0
    while start_idx < len(words):
        end_idx = min(start_idx + chunk_size, len(words))
        chunk = ' '.join(words[start_idx:end_idx])
        chunks.append(chunk.strip())
        start_idx += chunk_size - overlap
        if start_idx >= len(words):
            break
    
    return chunks

class Retrieval:
    def __init__(
        self,
        raw_doc_folder: str,
        encoded_doc_folder: str,
        model_name="BAAI/bge-m3"
    ):
        """
        Initialize Retrieval system with a SentenceTransformer model.
        
        Args:
            model_name: Name of the SentenceTransformer model (e.g., 'paraphrase-multilingual-MiniLM-L12-v2')
            raw_doc_folder: Folder containing raw PDF documents
            encoded_doc_folder: Folder to save encoded FAISS indexes and document texts
        """
        # Load SentenceTransformer model
        self.model = SentenceTransformer(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.raw_doc_folder = raw_doc_folder
        self.encoded_doc_folder = encoded_doc_folder
        self.indexes = {}
        self.documents = {}
        
        os.makedirs(self.encoded_doc_folder, exist_ok=True)

    def encode_document(
        self,
        doc_name: str,
        max_length: int = 512,
        batch_size: int = 8,
        force_reencode: bool = False,
        chunk_size: int = 300,
        chunk_overlap: int = 50
    ) -> None:
        """
        Encode a PDF document into chunked embeddings and store in a FAISS index.
        Each chunk starts with the page number from the PDF it primarily belongs to.
        """
        pdf_path = os.path.join(self.raw_doc_folder, f"{doc_name}.pdf")
        index_path = os.path.join(self.encoded_doc_folder, f"{doc_name}.bin")
        docs_path = os.path.join(self.encoded_doc_folder, f"{doc_name}.pkl")

        if not force_reencode and os.path.exists(index_path) and os.path.exists(docs_path):
            self.indexes[doc_name], self.documents[doc_name] = self._load_encoded_document(index_path, docs_path)
            return

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")

        # Extract text from PDF
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                page_texts = []
                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    text = page.extract_text()
                    if text:
                        text = text.strip()
                        if text:
                            page_texts.append((page_num, text))
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return

        if not page_texts:
            print(f"No valid text extracted from {pdf_path}")
            return

        # Chunk the text with page numbers
        documents = []
        current_words = []
        word_to_page = []  # List to track page number for each word

        for page_num, text in page_texts:
            words = text.split()
            # Assign page number to each word in this page
            word_to_page.extend([page_num] * len(words))
            current_words.extend(words)

            # Create chunks when we have enough words
            while len(current_words) >= chunk_size:
                chunk_words = current_words[:chunk_size]
                # Get the page number of the first word in the chunk
                chunk_page = word_to_page[0]
                chunk_text = f"\pagemark Page {chunk_page}: {' '.join(chunk_words)}"
                documents.append(chunk_text.strip())

                # Move forward by chunk_size - chunk_overlap
                current_words = current_words[chunk_size - chunk_overlap:]
                word_to_page = word_to_page[chunk_size - chunk_overlap:]

        # Handle remaining words
        if current_words:
            # Use the page number of the first word in the remaining chunk
            chunk_page = word_to_page[0] if word_to_page else page_texts[-1][0]
            chunk_text = f"\pagemark Page {chunk_page}: {' '.join(current_words)}"
            documents.append(chunk_text.strip())

        if not documents:
            print(f"No valid chunks created for {pdf_path}")
            return

        # Generate embeddings using SentenceTransformer
        embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=True,
            device=self.device,
            normalize_embeddings=True  # Normalize for cosine similarity
        ).astype(np.float32)

        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Use Inner Product (cosine similarity)
        faiss.normalize_L2(embeddings)  # Normalize embeddings for cosine similarity
        index.add(embeddings)

        # Save FAISS index and document chunks
        faiss.write_index(index, index_path)
        with open(docs_path, 'wb') as f:
            pickle.dump(documents, f)

        self.indexes[doc_name] = index
        self.documents[doc_name] = documents

    def _load_encoded_document(self, index_path: str, docs_path: str) -> Tuple[faiss.IndexFlatIP, List[str]]:
        index = faiss.read_index(index_path)
        with open(docs_path, 'rb') as f:
            documents = pickle.load(f)
        return index, documents

    def query_document(
        self,
        query: str,
        doc_name: str,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
        max_length: int = 512
    ) -> List[Tuple[str, float]]:
        """
        Query a specific document with a text query and return relevant text chunks.
        """
        if doc_name not in self.indexes or doc_name not in self.documents:
            raise ValueError(f"Document {doc_name} not found. Please encode it first.")

        # Encode query
        query_embedding = self.model.encode(
            [query],
            show_progress_bar=False,
            device=self.device,
            normalize_embeddings=True
        ).astype(np.float32)

        # Search FAISS index
        index = self.indexes[doc_name]
        documents = self.documents[doc_name]
        scores, indices = index.search(query_embedding, top_k)

        # Filter results by similarity threshold
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if score >= similarity_threshold and idx < len(documents):
                results.append(documents[idx])
        
        return results