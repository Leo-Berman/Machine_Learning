def main():
    file = open("algo_outputs.txt")
    rows = file.readlines()
    output_list = []
    for x in rows:
        scores = list(filter(lambda y:y.isdigit(),x.strip().split()))
        output_list.append(' '.join(scores)+"\n")

    file = open("formatted_shane_scores.txt",'w')
    file.writelines(output_list)
if __name__ == "__main__":
    main()