import json
import os
import logging
from typing import List, Dict
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalDataLoader:
    def __init__(self, data_dir: str = "data"):

        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
        logger.info(f"Initialized data loader for: {self.data_dir}")
    
    def load_json_file(self, file_path: Path) -> List[Dict]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                return data
            else:
                return [data]
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []
    
    def load_from_folder(self, folder_name: str) -> List[Dict]:

        folder_path = self.data_dir / folder_name
        
        if not folder_path.exists():
            logger.warning(f"Folder not found: {folder_path}")
            return []
        
        chunks = []
        json_files = list(folder_path.glob('*.json'))
        
        logger.info(f"Found {len(json_files)} JSON files in {folder_name}")
        
        for json_file in json_files:
            logger.info(f"Loading: {json_file.name}")
            file_chunks = self.load_json_file(json_file)
            chunks.extend(file_chunks)
        
        logger.info(f"Loaded {len(chunks)} chunks from {folder_name}")
        return chunks
    
    def load_handbook_data(self) -> List[Dict]:
        possible_names = ['handbook_paragraphs', 'handbook', 'debiased_data']
        
        for name in possible_names:
            chunks = self.load_from_folder(name)
            if chunks:
                return chunks
        
        logger.warning("No handbook data found")
        return []
    
    def load_transcript_data(self) -> List[Dict]:
        return self.load_from_folder('meeting_transcripts')
    
    def load_all_data(self) -> List[Dict]:
        logger.info("Loading all data...")
        
        handbook_chunks = self.load_handbook_data()
        transcript_chunks = self.load_transcript_data()
        
        all_chunks = handbook_chunks + transcript_chunks
        logger.info(f" Total chunks loaded: {len(all_chunks)}")
        
        return all_chunks
    
    def get_sample_chunk(self) -> Dict:
        all_chunks = self.load_all_data()
        if all_chunks:
            return all_chunks[0]
        return {}


if __name__ == "__main__":
    loader = LocalDataLoader(data_dir="data")
    
    chunks = loader.load_all_data()
    
    print(f"\n Total chunks loaded: {len(chunks)}")
    
    if chunks:
        print(f"\n Sample chunk structure:")
        sample = chunks[0]
        print(json.dumps(sample, indent=2)[:500] + "...")
        
        print(f"\n Keys in chunk: {list(sample.keys())}")