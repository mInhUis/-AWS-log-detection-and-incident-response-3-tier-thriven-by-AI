import json

file_path = 'data/raw/flaws_merged.jsonl'
max_results = 8
found_levels = 0
lines_scanned = 0

print(f"Scanning for the 'level' key in {file_path}...\n")

with open(file_path, 'r') as f:
    for line in f:
        lines_scanned += 1
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        # Explicitly check if "level" is a key in this JSON object
        if 'level' in evt:
            found_levels += 1
            lvl_value = evt['level']
            
            print(f"--- Found 'level' at line {lines_scanned} ---")
            print(f"  level value : {lvl_value!r}")
            # Printing eventName just to give you context of what the log actually is
            print(f"  eventName   : {evt.get('eventName', 'N/A')}") 
            print()
            
            if found_levels >= max_results:
                print(f"Reached limit of {max_results} found. Stopping.")
                break

if found_levels == 0:
    print(f"Scan complete. Checked {lines_scanned} lines.")
    print("Result: The exact key 'level' DOES NOT exist in any of these logs.")