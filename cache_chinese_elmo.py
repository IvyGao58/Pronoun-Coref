from elmoformanylangs import Embedder

e = Embedder('./elmo_chinese')

sents = [['今', '天', '天气', '真', '好', '阿'], ['潮水', '退', '了', '就', '知道', '谁', '沒', '起床']]

output = e.sents2elmo(sents)  # will return a list of numpy arrays each with the shape=(seq_len, embedding_size)

print(output)
