def main():
    file = open("picone_scores.txt")
    rows = file.readlines()
    output_list = []
    for x in rows:
        scores = list(filter(lambda y:y.isdigit(),x.strip().split()))[1::]
        error_sum = sum([int(x) for x in scores])
        output_list.append((' '.join(scores) + ' '+str(error_sum))+"\n")

    file = open("formatted_picone_scores.txt",'w')
    file.writelines(output_list)
if __name__ == "__main__":
    main()