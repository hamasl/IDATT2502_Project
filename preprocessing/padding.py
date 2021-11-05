import torch

def pad_and_mask(input):
	"""
	Input: A nested array containing tokenized functions
	Finds the largest tokenized function array, and makes the length a multiple of four
	Make each tokenized array the size of this number, and pads with 0, 
	to create a padded version of the tokenized array
	Creates a masked array to explain which values are padded

	returns padded array and masked array
	"""
	max_len_row = len(max(input, key=len))
	# Padded length must be multiple of 4
	max_len_row += (max_len_row % 4)
	padded = torch.zeros((len(input), max_len_row))

	for i in range(len(input)):
		current_row_length = len(input[i])
		padded[i, 0:current_row_length] = torch.tensor(input[i])
	masked = padded > 0
		
	return padded, masked
