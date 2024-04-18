import numpy as np
import scipy

def calculate_p_score(E1,E2,N):
    Z_score = (E1-E2)/np.sqrt((E1*(1-E1)/N) + (E2*(1-E2)/N))
    P_score = scipy.stats.norm.sf(Z_score)
    return P_score

def part_a(confidence,N):
    E1 = .2
    E2 = .19
    alpha = 1 - confidence
    p_score = calculate_p_score(E1,E2,N)
    print("\t\tE1 = {:.4f} E2 = {:.4f} N = {} P_score = {:.3f} Confidence = {:.2f} Statistical Significance = {}".format(E1,E2,N,p_score,confidence,p_score<alpha))

def part_b(confidence,N):
    E1 = .2
    E2 = .19
    alpha = 1 - confidence
    while(True):
        p_score = calculate_p_score(E1,E2,N)
       
        if p_score<alpha:
            print("\t\t Confidence = {:.2f} Min Decrease = {:.4f} N = {}".format(confidence,E1-E2,N))
            #P_score = {:.3f} Confidence = {:.2f} Statistical Significance = {}".format(E1-E2,N,p_score,confidence,p_score<alpha))
            break
        else:
            E2-=.0001

def main():
    print("Part a:")
    part_a(.8,1000)
    print("Part b")
    part_b(.8,1000)
    print("Part c")
    NS = [100,500,2000,5000,10000]
    CONFIDENCES = [.80,.85,.90,.95]
    for x in NS:
        for y in CONFIDENCES:
            #print("\tN = {} Confidence = {:.2f}".format(x,y))
            #part_a(y,x)
            part_b(y,x)
    
    
if __name__ == "__main__":
    main()
