def create_vocabulary(functions_list: [[]]):
    vocabulary = []
    for function in functions_list:
        for token in function:
            if token not in vocabulary: vocabulary.append(token)
    
    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
    idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

    return word2idx, idx2word