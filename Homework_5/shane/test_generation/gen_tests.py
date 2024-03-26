def main():
    hyp = open("hyps.txt","r")
    ref = open("refs.txt","r")

    hyp_lines = hyp.readlines()
    ref_lines = ref.readlines()

    hyp.close()
    ref.close()
    test_names = []
    for i in range(len(hyp_lines)):
        hyp_lines[i] = ' '.join(list(filter(lambda y: all("*" not in z for z in y),hyp_lines[i][6::].split())))+"\n"
        ref_lines[i] = ' '.join(list(filter(lambda y: all("*" not in z for z in y),ref_lines[i][6::].split())))
        file_name = "picone_test_"+str(i)+".txt"
        test_file = open(file_name,"w")
        test_file.writelines(hyp_lines[i])
        test_file.writelines(ref_lines[i])
        test_file.close()
        test_names.append(file_name+"\n")
        
    test_txt = open("test_names.txt","w")
    test_txt.writelines(test_names)
    test_txt.close()
if __name__ == "__main__":
    main()