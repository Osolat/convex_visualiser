import matplotlib.pyplot as plt
from os import walk
import numpy as np

def visualise():
    inputPointsX = []
    inputPointsXY = []
    hullX = []
    hullY = []
    lineNum = 0
    with open("visualiseFile") as file:
        for line in file:
            if lineNum == 0:
                points = line.split(";")
                points = points[:-1]
                for pointString in points:
                    var = pointString[pointString.find("(") + 1:pointString.find(")")]
                    xyvar = var.split(",")
                    inputPointsX.append(float(xyvar[0]))
                    inputPointsXY.append(float(xyvar[1]))
                lineNum += 1
                continue
            if lineNum == 1:
                points = line.split(";")
                points = points[:-1]
                for pointString in points:
                    var = pointString[pointString.find("(") + 1:pointString.find(")")]
                    xyvar = var.split(",")
                    hullX.append(float(xyvar[0]))
                    hullY.append(float(xyvar[1]))

    plt.scatter(inputPointsX, inputPointsXY, c="green")
    plt.scatter(hullX, hullY, c="r")
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    visualise()

