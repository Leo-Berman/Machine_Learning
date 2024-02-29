import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        prog='parseargs.py',
        description='Reads strings from a text file',
        epilog='Use as a function call in another script. Ex :\n<python3 scriptname -f filename'
        )
    parser.add_argument('-f','--filename')
    args = parser.parse_args()
    return args.filename

def read_txt():
    file_path = parse_args()
    with open(file_path,"r") as f:
        content = f.read().lower().split("\n")
        for i,x in enumerate(content):
            content[i] = x.split(" ")
    return content