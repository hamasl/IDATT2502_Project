import numpy as np

def index_pairing(function_list, word2idx_dict, window_size = 2):
	idx_pairs = []

	for function in function_list:
		indices = [word2idx_dict[word] for word in function]

		for center_word_pos in range(len(indices)):
			for w in range(-window_size, window_size + 1):
				context_word_pos = center_word_pos + w

				if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
					continue

				idx_pairs.append((indices[center_word_pos], indices[context_word_pos]))

	return np.array(idx_pairs)
