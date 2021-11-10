import numpy as np
import vocabulary as vocabulary

def index_pairing(function_list, word2idx_dict, window_size = 4):
	idx_pairs = []

	for function in function_list:
		indices = [word2idx[word] for word in function]

		for center_word_pos in range(len(indices)):
			for w in range(-window_size, window_size + 1):
				context_word_pos = center_word_pos + w

				if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
					continue

				idx_pairs.append((indices[center_word_pos], indices[context_word_pos]))

	return np.array(idx_pairs)

					
# if __name__ =='__main__':
# 	functions_list = [['hey', 'cunt', 'hvor', 'er', 'du', 'hen'], ['abc', 'bcd']]
# 	word2idx, idx2word = vocabulary.create_vocabulary(functions_list)
# 	print(len(index_pairing(functions_list, word2idx, window_size=20)))