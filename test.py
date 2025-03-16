import time
from steganography import embed_text_lsb

image_path = r"static\uploads\ACSS Bulletin Board Layout.png"
save_folder = "test/output"

text_lengths = [19,19,19,255,255,32,50,100,50,100,4,30,2,30,2,20,1] 
results = []

for length in text_lengths:
    text = "A" * length  
    start_time = time.time()
    embed_text_lsb(image_path, text, save_folder)
    end_time = time.time()

    embedding_time = (end_time - start_time) * 1000  
    results.append((length, embedding_time))
    print(f"Text Length: {length}, Embedding Time: {embedding_time:.2f} ms")

average_time = sum(time for _, time in results) / len(results)
print(f"\nAverage Embedding Time: {average_time:.2f} ms per image")
#Average embedding time resulted to 46.35 ms per image