import io
import zipfile
import requests
import frontmatter
import os
from minsearch import Index
import numpy as np
from minsearch import VectorSearch
from ultralytics_embeddings import generate_embeddings

ultralytics_embeddings = np.loadtxt("ultralyticembeddings.txt")

def read_repo_data(repo_owner, repo_name):
    repo_identifier = f"{repo_owner}/{repo_name}"
    repository_data = []
    
    # Check if repo is ultralytics/ultralytics and use local folder
    if repo_identifier == "ultralytics/ultralytics":
        ultralytics_folder = os.path.join(os.path.dirname(__file__), "ultralytics_files")
        print(f"Reading local files from: {ultralytics_folder}")
        
        if not os.path.exists(ultralytics_folder):
            raise FileNotFoundError(f"Ultralytics folder not found at: {ultralytics_folder}")
        
        for filename in os.listdir(ultralytics_folder):
            if filename.lower().endswith(('.md', '.mdx')):
                file_path = os.path.join(ultralytics_folder, filename)
                with open(file_path, 'rb') as f_in:
                    content = f_in.read()
                    post = frontmatter.loads(content)
                    data = post.to_dict()
                    data['filename'] = filename
                    repository_data.append(data)
    else:
        print(f"Downloading repository: {repo_identifier}")
        url = f'https://codeload.github.com/{repo_owner}/{repo_name}/zip/refs/heads/main'
        resp = requests.get(url)
        zf = zipfile.ZipFile(io.BytesIO(resp.content))

        for file_info in zf.infolist():
            filename = file_info.filename.lower()

            if not (filename.endswith('.md') or filename.endswith('.mdx')):
                continue

            with zf.open(file_info) as f_in:
                content = f_in.read()
                post = frontmatter.loads(content)
                data = post.to_dict()

                _, filename_repo = file_info.filename.split('/', maxsplit=1)
                data['filename'] = filename_repo
                repository_data.append(data)

        zf.close()

    return repository_data

def sliding_window(seq, size, step):
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    n = len(seq)
    result = []
    for i in range(0, n, step):
        batch = seq[i:i+size]
        result.append({'start': i, 'content': batch})
        if i + size > n:
            break

    return result


def chunk_documents(docs, size=2000, step=1000):
    chunks = []

    for doc in docs:
        doc_copy = doc.copy()
        doc_content = doc_copy.pop('content')
        doc_chunks = sliding_window(doc_content, size=size, step=step)
        for chunk in doc_chunks:
            chunk.update(doc_copy)
        chunks.extend(doc_chunks)

    return chunks


def index_data(
    repo_owner,
    repo_name,
    chunk_size=2000,
    chunk_step=1000,
    vector= False ):
   # Read all markdown documents from the repo
    docs = read_repo_data(repo_owner, repo_name)
    
    # Chunk the documents using the specified size and step
    doc_chunks = chunk_documents(docs, size=chunk_size, step=chunk_step)
    
    # Add a comment to indicate if chunking was successful
    if len(doc_chunks) == 0:
        print("Warning: No document chunks were created. Check if the documents have content.")
    else:
        print(f"Chunking successful: {len(doc_chunks)} chunks created from {len(docs)} documents.")

    # Create and fit the index
    index = Index(text_fields=["chunk", "title", "description", "filename"], keyword_fields=[])
    index.fit(doc_chunks)

    if vector:
        repo_identifier = f"{repo_owner}/{repo_name}"
        if repo_identifier == "ultralytics/ultralytics":
            doc_embeddings = np.loadtxt("ultralyticembeddings.txt")
        else:
            doc_embeddings = generate_embeddings(chunks=doc_chunks)

        print(f"ultralytics_embeddings shape: {doc_embeddings.shape}")
        vindex = VectorSearch()
        vindex.fit(doc_embeddings, doc_chunks)  
        return index, vindex
    else:
        return index

# if __name__ == "__main__":
#     REPO_OWNER = "ultralytics"
#     REPO_NAME = "ultralytics"


#     index, vindex = index_data(REPO_OWNER, REPO_NAME, vector=True)
#     print(index)
#     print("Indexing complete.")