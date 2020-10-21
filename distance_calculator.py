from scipy.spatial import distance as dist

#variable Definition
imageWidth=0.955
IV = [[1, 0.5], [0.5, 1]]

#function definition

def distance(xA,yA,xB,yB):
    D = dist.mahalanobis([xA, yA], [xB, yB],IV)
    return D

