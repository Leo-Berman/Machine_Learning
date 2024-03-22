import sys
import numpy as np

def align_strings(ref, hyp):

    # Convert strings to lowercase and split into lists of words
    ref = ref.lower().split()
    hyp = hyp.lower().split()

    # Initialize dynamic programming matrix
    dp = np.array([[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)])

    # Fill the dynamic programming matrix
    for i in range(len(ref) + 1):
        for j in range(len(hyp) + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])


    print(dp)

    # Backtrack to find the aligned strings
    i, j = len(ref), len(hyp)
    aligned_ref = []
    aligned_hyp = []
    subs = ins = dels = 0

    # iterate over the entire matrix from its last position
    while i > 0 or j > 0:

        # check if the words are the same
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            aligned_ref.append(ref[i - 1])
            aligned_hyp.append(hyp[j - 1])
            i -= 1
            j -= 1

        # check if the words are different
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            aligned_ref.append(ref[i - 1].upper())
            aligned_hyp.append(hyp[j - 1].upper())
            subs += 1
            i -= 1
            j -= 1

        # check if a words has been removed
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            aligned_ref.append(ref[i - 1].upper())
            aligned_hyp.append(len(ref[i - 1])*"*")
            dels += 1
            i -= 1

        # check if a word has been added
        else:
            aligned_ref.append(len(hyp[j - 1])*"*")
            aligned_hyp.append(hyp[j - 1].upper())
            ins += 1
            j -= 1

    # reverse the strings to put them pack into the correct sequence
    aligned_ref.reverse()
    aligned_hyp.reverse()

    return aligned_ref, aligned_hyp, subs, ins, dels

def test(fname, *, verbose=True):

    # Read content from file
    with open(fname, "r") as file:
        content = file.readlines()

    # get the reference string
    try:
        ref = content[0].strip()
    except IndexError:
        ref = ""

    # get the hypothesis string 
    try:   
        hyp = content[1].strip()
    except IndexError:
        hyp = ""

    # Align strings
    aligned_ref, aligned_hyp, subs, ins, dels = align_strings(ref, hyp)

    # Print aligned strings and error statistics
    if verbose:
        print("R:", " ".join(aligned_ref))
        print("H:", " ".join(aligned_hyp))
        print("Subs:", subs, "Ins:", ins, "Dels:", dels, "Total Errors:", subs + ins + dels)

    return aligned_ref, aligned_hyp, subs, ins, dels


def main(argv):

    for fname in argv[1:]:
        print(f"{fname.upper()}:")
        test(fname)
        print()

if __name__ == "__main__":
    main(sys.argv)
