import json
import os
import logging
from typing import List, Dict
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalDataLoader:
    """
    Loads data from local storage for the Intelligent Onboarding Assistant.
    Supports multiple data sources: handbook files, debiased data, and meeting transcripts.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the data directory (default: "data")
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
        logger.info(f"✅ Initialized data loader for: {self.data_dir}")
    
    def load_json_file(self, file_path: Path) -> List[Dict]:
        """
        Load a single JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of dictionaries (chunks)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Ensure we always return a list
            if isinstance(data, list):
                return data
            else:
                return [data]
                
        except Exception as e:
            logger.error(f"❌ Error loading {file_path}: {e}")
            return []
    
    def load_from_folder(self, folder_name: str) -> List[Dict]:
        """
        Load all JSON files from a folder.
        
        Args:
            folder_name: Name of the folder within data_dir
            
        Returns:
            List of dictionaries (chunks) from all JSON files
        """
        folder_path = self.data_dir / folder_name
        
        if not folder_path.exists():
            logger.debug(f"Folder not found: {folder_path}")
            return []
        
        chunks = []
        json_files = list(folder_path.glob('*.json'))
        
        if not json_files:
            logger.debug(f"No JSON files found in {folder_name}")
            return []
        
        logger.info(f"📁 Found {len(json_files)} JSON files in {folder_name}/")
        
        for json_file in json_files:
            logger.info(f"   Loading: {json_file.name}")
            file_chunks = self.load_json_file(json_file)
            chunks.extend(file_chunks)
        
        logger.info(f"✅ Loaded {len(chunks)} chunks from {folder_name}/")
        return chunks
    
    def load_handbook_data(self) -> List[Dict]:
        """
        Load handbook data from available sources.
        Priority order:
        1. debiased_data/handbook_paragraphs_debiased.json (best quality)
        2. handbook_paragraphs.json (root level)
        3. debiased_data/ folder (all files)
        4. Other fallback locations
        
        Returns:
            List of handbook chunks
        """
        logger.info("📚 Loading handbook data...")
        
        # PRIORITY 1: Debiased handbook (best quality)
        debiased_handbook = self.data_dir / 'debiased_data' / 'handbook_paragraphs_debiased.json'
        if debiased_handbook.exists():
            logger.info(f"   Using: debiased_data/handbook_paragraphs_debiased.json")
            chunks = self.load_json_file(debiased_handbook)
            if chunks:
                logger.info(f"✅ Loaded {len(chunks)} chunks from debiased handbook")
                return chunks
        
        # PRIORITY 2: Root level handbook file
        root_handbook = self.data_dir / 'handbook_paragraphs.json'
        if root_handbook.exists():
            logger.info(f"   Using: handbook_paragraphs.json")
            chunks = self.load_json_file(root_handbook)
            if chunks:
                logger.info(f"✅ Loaded {len(chunks)} chunks from root handbook")
                return chunks
        
        # PRIORITY 3: All files in debiased_data folder
        debiased_folder = self.data_dir / 'debiased_data'
        if debiased_folder.exists() and debiased_folder.is_dir():
            logger.info(f"   Using: debiased_data/ folder")
            chunks = self.load_from_folder('debiased_data')
            if chunks:
                return chunks
        
        # FALLBACK: Other possible locations
        for folder_name in ['handbook_paragraphs', 'handbook', 'processed']:
            folder_path = self.data_dir / folder_name
            if folder_path.exists() and folder_path.is_dir():
                logger.info(f"   Trying fallback: {folder_name}/")
                chunks = self.load_from_folder(folder_name)
                if chunks:
                    return chunks
        
        logger.warning("⚠️ No handbook data found!")
        return []
    
    def load_transcript_data(self) -> List[Dict]:
        """
        Load meeting transcript data.
        Priority order:
        1. debiased_data/all_transcripts_debiased.json (best quality)
        2. meeting_transcripts/all_transcripts.json
        3. meeting_transcripts/ folder (all files)
        
        Returns:
            List of transcript chunks
        """
        logger.info("🎤 Loading meeting transcripts...")
        
        # PRIORITY 1: Debiased transcripts (best quality)
        debiased_transcripts = self.data_dir / 'debiased_data' / 'all_transcripts_debiased.json'
        if debiased_transcripts.exists():
            logger.info(f"   Using: debiased_data/all_transcripts_debiased.json")
            chunks = self.load_json_file(debiased_transcripts)
            if chunks:
                logger.info(f"✅ Loaded {len(chunks)} transcript chunks (debiased)")
                return chunks
        
        # PRIORITY 2: Original transcripts file
        original_transcripts = self.data_dir / 'meeting_transcripts' / 'all_transcripts.json'
        if original_transcripts.exists():
            logger.info(f"   Using: meeting_transcripts/all_transcripts.json")
            chunks = self.load_json_file(original_transcripts)
            if chunks:
                logger.info(f"✅ Loaded {len(chunks)} transcript chunks")
                return chunks
        
        # PRIORITY 3: All files in meeting_transcripts folder
        transcripts_folder = self.data_dir / 'meeting_transcripts'
        if transcripts_folder.exists() and transcripts_folder.is_dir():
            logger.info(f"   Using: meeting_transcripts/ folder")
            return self.load_from_folder('meeting_transcripts')
        
        logger.warning("⚠️ No transcript data found!")
        return []
    
    def load_all_data(self) -> List[Dict]:
        """
        Load all available data (handbook + transcripts).
        Uses debiased versions when available for best quality.
        
        Returns:
            Combined list of all chunks
        """
        logger.info("="*60)
        logger.info("📊 Loading all data sources...")
        logger.info("="*60)
        
        handbook_chunks = self.load_handbook_data()
        transcript_chunks = self.load_transcript_data()
        
        all_chunks = handbook_chunks + transcript_chunks
        
        logger.info("="*60)
        logger.info(f"✅ Total chunks loaded: {len(all_chunks)}")
        logger.info(f"   - Handbook: {len(handbook_chunks)} chunks")
        logger.info(f"   - Transcripts: {len(transcript_chunks)} chunks")
        logger.info("="*60)
        
        return all_chunks
    
    def load_debiased_data_only(self) -> List[Dict]:
        """
        Load ONLY debiased data (highest quality).
        Use this for production/final evaluation.
        
        Returns:
            List of debiased chunks
        """
        logger.info("🔬 Loading debiased data only...")
        
        all_chunks = []
        
        # Load debiased handbook
        debiased_handbook = self.data_dir / 'debiased_data' / 'handbook_paragraphs_debiased.json'
        if debiased_handbook.exists():
            chunks = self.load_json_file(debiased_handbook)
            all_chunks.extend(chunks)
            logger.info(f"   Handbook: {len(chunks)} chunks")
        
        # Load debiased transcripts
        debiased_transcripts = self.data_dir / 'debiased_data' / 'all_transcripts_debiased.json'
        if debiased_transcripts.exists():
            chunks = self.load_json_file(debiased_transcripts)
            all_chunks.extend(chunks)
            logger.info(f"   Transcripts: {len(chunks)} chunks")
        
        logger.info(f"✅ Total debiased chunks: {len(all_chunks)}")
        return all_chunks
    
    def get_sample_chunk(self) -> Dict:
        """
        Get a sample chunk for testing/inspection.
        
        Returns:
            First chunk from the dataset, or empty dict if no data
        """
        all_chunks = self.load_all_data()
        if all_chunks:
            return all_chunks[0]
        return {}
    
    def get_data_stats(self) -> Dict:
        """
        Get statistics about the loaded data.
        
        Returns:
            Dictionary with data statistics
        """
        all_chunks = self.load_all_data()
        
        if not all_chunks:
            return {
                'total_chunks': 0,
                'sources': {},
                'average_text_length': 0
            }
        
        # Count chunks by source
        sources = {}
        total_text_length = 0
        
        for chunk in all_chunks:
            source = chunk.get('source', chunk.get('type', 'unknown'))
            sources[source] = sources.get(source, 0) + 1
            
            # Calculate text length
            text = chunk.get('text', '') or chunk.get('paragraph', '') or chunk.get('content', '')
            total_text_length += len(text)
        
        return {
            'total_chunks': len(all_chunks),
            'sources': sources,
            'average_text_length': total_text_length // len(all_chunks) if all_chunks else 0,
            'sample_keys': list(all_chunks[0].keys()) if all_chunks else []
        }


if __name__ == "__main__":
    """
    Test the data loader by loading all data and displaying statistics.
    """
    print("\n" + "="*80)
    print("  LOCAL DATA LOADER - TEST RUN")
    print("="*80 + "\n")
    
    # Initialize loader
    loader = LocalDataLoader(data_dir="data")
    
    # Load all data
    print("\n🔹 OPTION 1: Load ALL available data")
    print("-"*80)
    chunks = loader.load_all_data()
    print(f"Total chunks: {len(chunks)}")
    
    # Load only debiased data
    print("\n🔹 OPTION 2: Load ONLY debiased data")
    print("-"*80)
    debiased_chunks = loader.load_debiased_data_only()
    print(f"Total debiased chunks: {len(debiased_chunks)}")
    
    if chunks:
        # Display sample chunk
        print(f"\n📄 SAMPLE CHUNK STRUCTURE:")
        print("-"*80)
        sample = chunks[0]
        print(json.dumps(sample, indent=2)[:500] + "...")
        
        print(f"\n🔑 KEYS IN CHUNK:")
        print("-"*80)
        print(f"Keys: {list(sample.keys())}")
        
        # Display statistics
        print(f"\n📈 DATA STATISTICS:")
        print("-"*80)
        stats = loader.get_data_stats()
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Average text length: {stats['average_text_length']} characters")
        print(f"\nSources breakdown:")
        for source, count in stats['sources'].items():
            print(f"  - {source}: {count} chunks")
    else:
        print("\n⚠️ No data loaded!")
    
    print("\n" + "="*80 + "\n")