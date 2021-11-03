import torch

def masking_padding(input):
	max_len_row = len(max(input, key=len))
	padded = torch.zeros((len(input), max_len_row))

	for i in range(len(input)):
		current_row_length = len(input[i])
		padded[i, 0:current_row_length] = torch.tensor(input[i])
	masked = padded > 0
		
	return padded, masked
