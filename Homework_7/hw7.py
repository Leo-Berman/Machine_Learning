import pandas as pd

def find_col_range(incol:list):
    return max(incol) - min(incol)

def quantize_list(incol:list):
    list_range = find_col_range(incol)
    ret_list = []
    for x in incol:
        ret_list.append(round((x/list_range)*128))
    
    return ret_list
def main():

    # read data in
    #
    header = ["class","x_coord","y_coord"]
    traindf = pd.read_csv("train.csv",names = header)
    devdf = pd.read_csv("dev.csv",names = header)

    # convert 2d coords to lists
    #
    train_xcoords = traindf["x_coord"]
    train_ycoords = traindf["y_coord"]
    dev_xcoords = devdf["x_coord"]
    dev_ycoords = devdf["y_coord"]

    # quantize lists
    #
    quantized_train_xcoords = quantize_list(train_xcoords)
    quantized_train_ycoords = quantize_list(train_ycoords)
    quantized_dev_xcoords = quantize_list(dev_xcoords)
    quantized_dev_ycoords = quantize_list(dev_ycoords)

    

    print(quantized_train_xcoords)


if __name__ == "__main__":
    main()