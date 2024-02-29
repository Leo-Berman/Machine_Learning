from helperfunctions import parsetxt
from helperfunctions import comparewords

def main():
    
    # read the text file and parse the first two lines
    # as a sentence each
    content = parsetxt.read_txt()
    sentence1 = content[0]
    sentence2 = content[1]

    # create a list that will contain the matched sentence
    matched_sentence = []
    
    # create a list for words that have been used
    used_words = []

    # iterate through each word in sentence one
    for i,x in enumerate(sentence1):
        print(x)
        possible_words = []
        # iterate through each word in sentence 2
        for j,y in enumerate(sentence2):
            score = comparewords.compare_words(x,y)
            if y not in used_words:
                possible_words.append([y,score])
        possible_words.sort(key=lambda x: x[1])
        print(possible_words)

main()