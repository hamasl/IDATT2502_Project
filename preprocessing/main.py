from preprocessing.tokenizer import tokenize
import preprocessing.keyword_dictionary as keyword_dictionary
import preprocessing.generalizer as generalizer

if __name__ == '__main__':
    x, y = tokenize()
    dictionary = keyword_dictionary.get()
    x = generalizer.handle_functions_and_variables(generalizer.handle_literals(x, dictionary), dictionary)
    print(x, y)

