import re
from stopwords import *


# Returns dictionary of word counts for a text
def get_word_counts(text, all_words):
    wc = {}
    words = get_words(text)
    # Loop over all the entries

    for word in words:
        if (word not in stopwords) and (word in all_words):
            wc[word] = wc.get(word, 0)+1

    return wc


# splits text into words
def get_words(txt):
    # Split words by all non-alpha characters
    words = re.compile(r'[^A-Z^a-z]+').split(txt)

    # Convert to lowercase
    return [word.lower() for word in words if word != '']


# converts counts into a vector
def get_word_vector(word_list, wc):
    v = [0]*len(word_list)
    for i in range(len(word_list)):
        if word_list[i] in wc:
            v[i] = wc[word_list[i]]
    return v


# prints matrix
def print_word_matrix(docs):
    for d in docs:
        print(d[0], d[1])


if __name__ == '__main__':
    file_name = "dataset.csv"
    f = open(file_name, "r", encoding="utf-8")
    out = open(file_name.split('.')[0] + "_vectors.txt", "w")

    docs = []
    lines = [line for line in f]
    for i in range(len(lines)):
        buffer = lines[i].split(',')
        docs.append(buffer)
        length = len(buffer)
        docs[i][length - 1] = docs[i][length - 1][0:len(buffer[length - 1]) - 1]
    # print(docs)

    for items in docs[0]:
        out.write('\t' + items)
    out.write('\n')

    for index in range(len(docs)):
        if index == 0:
            continue
        else:
            out.write('d' + str(index) + '\t')
            for items in docs[index]:
                out.write(items + '\t')
            out.write('\n')






