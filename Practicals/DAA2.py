#2. Write a program to implement Huffman Encoding using a greedy strategy.


import heapq
from collections import defaultdict, Counter

# Node class to represent each character and its frequency
class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    # Define comparison operators for priority queue
    def __lt__(self, other):
        return self.freq < other.freq

# Function to build Huffman Tree using the frequency data
def build_huffman_tree(char_freq):
    # Create a priority queue with initial nodes for each character
    heap = [HuffmanNode(char, freq) for char, freq in char_freq.items()]
    heapq.heapify(heap)

    # Merge nodes until only one node (the root of the Huffman Tree) remains
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

# Function to generate Huffman Codes from the tree
def generate_huffman_codes(root, current_code="", codes={}):
    if root is None:
        return

    # Leaf node, store the code
    if root.char is not None:
        codes[root.char] = current_code

    # Traverse left and right
    generate_huffman_codes(root.left, current_code + "0", codes)
    generate_huffman_codes(root.right, current_code + "1", codes)

# Main function to encode data using Huffman Encoding
def huffman_encoding(data):
    if not data:
        return "", {}

    # Calculate frequency of each character
    char_freq = Counter(data)

    # Build Huffman Tree
    root = build_huffman_tree(char_freq)

    # Generate Huffman Codes
    codes = {}
    generate_huffman_codes(root, "", codes)

    # Encode the data
    encoded_data = "".join(codes[char] for char in data)

    return encoded_data, codes

# Main function to decode data using Huffman Encoding
def huffman_decoding(encoded_data, codes):
    # Build a reverse mapping of the codes
    reverse_codes = {v: k for k, v in codes.items()}
    decoded_data = ""
    current_code = ""

    for bit in encoded_data:
        current_code += bit
        if current_code in reverse_codes:
            decoded_data += reverse_codes[current_code]
            current_code = ""

    return decoded_data

# Example usage
if __name__ == "__main__":
    data = "huffman encoding example"
    print("Original Data:", data)

    # Encoding
    encoded_data, codes = huffman_encoding(data)
    print("\nEncoded Data:", encoded_data)
    print("Huffman Codes:", codes)

    # Decoding
    decoded_data = huffman_decoding(encoded_data, codes)
    print("\nDecoded Data:", decoded_data)
