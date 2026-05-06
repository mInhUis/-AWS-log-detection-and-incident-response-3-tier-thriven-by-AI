import json
import glob
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Final, Iterator

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent



def retrieve_all_reports(directory_path):
    """
    Scans a directory for JSON files and extracts the 'report_text' value.
    Returns a dictionary: { 'filename': 'report_content' }
    """
    report_data = {}
    
    # Create the search pattern (e.g., ../reports/*.json)
    search_pattern = os.path.join(directory_path, "*.json")
    
    for file_path in glob.glob(search_pattern):
        filename = os.path.basename(file_path)
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Using .get() prevents the script from crashing if the key is missing
                text = data.get("report_text", "Key 'report_text' not found")
                report_data[filename] = text
                
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {filename}: {e}")
            
    return report_data

def main():
    path = _PROJECT_ROOT
    retrieve_all_reports(path)

if __name__ == "__main__":
    main()
# Example usage:
# path = "../reports"
# results = retrieve_all_reports(path)
# for file, text in results.items():
#     print(f"--- {file} ---\n{text}\n")