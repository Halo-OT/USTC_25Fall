import os
import json
import time
from storage.hbase_client import HBaseClient
from storage.data_model import Document

def import_data_to_hbase():
    print("Connecting to HBase...")
    client = HBaseClient()
    
    if not client.use_hbase:
        print("Error: Could not connect to HBase. Please check if HBase is running.")
        return

    data_dir = os.path.join(os.path.dirname(__file__), 'data/files')
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return

    print(f"Importing data from {data_dir} to HBase...")
    count = 0
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(data_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                doc = Document.from_dict(data)
                row_key = client.save_document(doc)
                print(f"Imported {filename} -> {row_key}")
                count += 1
            except Exception as e:
                print(f"Failed to import {filename}: {e}")

    print(f"Successfully imported {count} documents to HBase.")
    client.close()

if __name__ == "__main__":
    import_data_to_hbase()



