import torch

def masking_padding(input):
	max_len_row = len(max(input, key=len))
	# Padded length must be multiple of 4
	max_len_row += (max_len_row % 4)
	padded = torch.zeros((len(input), max_len_row))

	for i in range(len(input)):
		current_row_length = len(input[i])
		padded[i, 0:current_row_length] = torch.tensor(input[i])
	masked = padded > 0
		
	return padded, masked
