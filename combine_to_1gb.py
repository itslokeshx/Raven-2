import os
import sys
from tqdm import tqdm

def main():
    dest_path = "data/combined.txt"
    os.makedirs("data", exist_ok=True)
    
    # 1. Look for candidate files
    search_dirs = [".", "data"]
    variations = ["corpus_{}.txt", "corpus{}.txt", "corpus {}.txt", "corpus-{}.txt"]
    candidates = []
    
    for i in range(1, 5):
        found = False
        for sdir in search_dirs:
            for var in variations:
                path = os.path.join(sdir, var.format(i))
                if os.path.exists(path):
                    candidates.append(path)
                    found = True
                    break
            if found:
                break
                
    if not candidates:
        print("❌ No source corpus files found to combine!")
        sys.exit(1)
        
    candidates = sorted(list(set(candidates)))
    
    print("=" * 60)
    print(" 🛠️  RAVEN-2 TOTAL CORPUS GENERATOR")
    print(f" Source Files: {candidates}")
    print(f" Destination:  {dest_path}")
    print("=" * 60)
    
    buffer_size = 10 * 1024 * 1024  # 10MB chunk
    
    with open(dest_path, "w", encoding="utf-8") as out_file:
        for filepath in candidates:
            file_size = os.path.getsize(filepath)
            print(f"\nProcessing {filepath} ({file_size / (1024**2):.2f} MB)...")
            
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f, tqdm(
                total=file_size, unit='B', unit_scale=True, desc=os.path.basename(filepath)
            ) as pbar:
                while True:
                    chunk = f.read(buffer_size)
                    if not chunk:
                        break
                    pbar.update(len(chunk.encode("utf-8")))
                    out_file.write(chunk)
                        
    final_size_mb = os.path.getsize(dest_path) / (1024**2)
    print("\n" + "=" * 60)
    print(" 🎉 TOTAL CORPUS MERGE COMPLETED SUCCESSFULLY!")
    print(f" Final File: {dest_path}")
    print(f" Final Size: {final_size_mb:.2f} MB ({final_size_mb / 1024:.3f} GB)")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
