
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-2")
print(encoding.encode("The quick brown fox jumps over the lazy dog."))

print(encoding.decode([464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13]))

print(encoding.decode([464]))
print(encoding.decode([2068]))
print(encoding.decode([13]))
