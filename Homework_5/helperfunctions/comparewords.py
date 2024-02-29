# function for comparing two words for 
# similarities
#
def compare_words(word1,word2,display=False):

    # find the length of each word
    word1_length = len(word1)
    word2_length = len(word2)

    # find which word is longer
    if word1_length > word2_length:
        longer_word = [*word1]
        longer_word_length = word1_length
        shorter_word = [*word2]
        shorter_word_length = word2_length
    else:
        longer_word = [*word2]
        longer_word_length = word2_length
        shorter_word = [*word1]
        shorter_word_length = word1_length

    # number to keep track of difference score
    score = 0
    
    # fill length difference in
    length_difference = longer_word_length - shorter_word_length

    # add the deletions/insertions into the score
    score+=length_difference

    # fill in for the length difference
    [shorter_word.insert(shorter_word_length//2,"*") for x in range(length_difference)]

    if display == True:
        print("Longer word  = ",longer_word)
        print("Shorter word = ",shorter_word)
    # length of both lists after the fill of "**"
    comparison_length = longer_word_length

    # iterate through the words and add to the score for
    # each mismatched letter
    for i in range(comparison_length):
        if shorter_word[i] != longer_word[i]:
            score += 1
    
    return score

