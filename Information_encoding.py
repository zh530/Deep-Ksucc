import numpy as np


# 说明： one of K编码
# 输入： data
# 输出： data_X, data_Y
def one_hot(data, windows=16):
    # define input string
    data = data
    length = len(data)
    # define empty array
    data_X = np.zeros((length, 2*windows+1, 21))
    # print(data_X)
    data_Y = []
    for i in range(length):
        x = data[i].split()
        # get label
        data_Y.append(int(x[1]))
        # print(data_Y)
        # define universe of possible input values
        alphabet = 'ACDEFGHIKLMNPQRSTVWY-BJOUXZ'
        # define a mapping of chars to integers
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        # integer encode input data
        integer_encoded = [char_to_int[char] for char in x[2]]
        # one hot encode
        j = 0
        for value in integer_encoded:
            if value in [21, 22, 23, 24, 25, 26]:
                for k in range(21):
                    data_X[i][j][k] = 0.05
            else:
                data_X[i][j][value] = 1.0
            j = j + 1
    data_Y = np.array(data_Y)
    # print(data_X)

    return data_X, data_Y


# 说明： 氨基酸理化信息编码
# 输入： data
# 输出： data_X
# 来源： 2019年论文“TOXIFY: a deep learning approach to classify animal venom proteins”,总结了500个AA属性，这些属性反应了极性、二级结构、分子体积、密码子多样性和静电荷
def Phy_Chem_Inf(data, windows=16):
    letterDict = {}
    letterDict["A"] = [-0.591, -1.302, -0.733, 1.570, -0.146]
    letterDict["C"] = [-1.343, 0.465, -0.862, -1.020, -0.255]
    letterDict["D"] = [1.050, 0.302, -3.656, -0.259, -3.242]
    letterDict["E"] = [1.357, -1.453, 1.477, 0.113, -0.837]
    letterDict["F"] = [-1.006, -0.590, 1.891, -0.397, 0.412]
    letterDict["G"] = [-0.384, 1.652, 1.330, 1.045, 2.064]
    letterDict["H"] = [0.336, -0.417, -1.673, -1.474, -0.078]
    letterDict["I"] = [-1.239, -0.547, 2.131, 0.393, 0.816]
    letterDict["K"] = [1.831, -0.561, 0.533, -0.277, 1.648]
    letterDict["L"] = [-1.019, -0.987, -1.505, 1.266, -0.912]
    letterDict["M"] = [-0.663, -1.524, 2.219, -1.005, 1.212]
    letterDict["N"] = [0.945, 0.828, 1.299, -0.169, 0.933]
    letterDict["P"] = [0.189, 2.081, -1.628, 0.421, -1.392]
    letterDict["Q"] = [0.931, -0.179, -3.005, -0.503, -1.853]
    letterDict["R"] = [1.538, -0.055, 1.502, 0.440, 2.897]
    letterDict["S"] = [-0.228, 1.399, -4.760, 0.670, -2.647]
    letterDict["T"] = [-0.032, 0.326, 2.213, 0.908, 1.313]
    letterDict["V"] = [-1.337, -0.279, -0.544, 1.242, -1.262]
    letterDict["W"] = [-0.595, 0.009, 0.672, -2.128, -0.184]
    letterDict["Y"] = [0.260, 0.830, 3.097, -0.838, 1.512]
    letterDict["-"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["B"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["J"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["O"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["U"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["X"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["Z"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    """
    letterDict = {"A": [-0.59, -1.30, -0.73, 1.57, -0.15],
                  "C": [-1.34, 0.47, -0.86, -1.02, -0.26],
                  "D": [1.05, 0.30, -3.66, -0.26, -3.24],
                  "E": [1.36, -1.45, 1.48, 0.11, -0.84],
                  "F": [-1.01, -0.59, 1.89, -0.40, 0.41],
                  "G": [-0.38, 1.65, 1.33, 1.05, 2.06],
                  "H": [0.34, -0.42, -1.67, -1.47, -0.08],
                  "I": [-1.24, -0.55, 2.13, 0.39, 0.82],
                  "K": [1.83, -0.56, 0.53, -0.28, 1.65],
                  "L": [-1.02, -0.99, -1.51, 1.27, -0.91],
                  "M": [-0.66, -1.52, 2.22, -1.01, 1.21],
                  "N": [0.95, 0.83, 1.30, -0.17, 0.93],
                  "P": [0.19, 2.08, -1.63, 0.42, -1.39],
                  "Q": [0.93, -0.18, -3.01, -0.50, -1.85],
                  "R": [1.54, -0.06, 1.50, 0.44, 2.90],
                  "S": [-0.23, 1.40, -4.76, 0.67, -2.65],
                  "T": [-0.03, 0.33, 2.21, 0.91, 1.31],
                  "V": [-1.34, -0.28, -0.54, 1.24, -1.26],
                  "W": [-0.60, 0.01, 0.67, -2.13, -0.18],
                  "Y": [0.26, 0.83, 3.10, -0.84, 1.51],
                  "-": [0.0, 0.0, 0.0, 0.0, 0.0],
                  "B": [1, 0.565, -1.18, -0.215, -1.155],
                  "J": [-1.13, -0.77, 0.31, 0.83, -0.045],
                  "O": [-0.13, -0.12, 0.6, -0.03, -0.115],
                  "U": [-0.13, -0.12, 0.6, -0.03, -0.115],
                  "X": [-0.13, -0.12, 0.6, -0.03, -0.115],
                  "Z": [1.145, -0.815, -0.765, -0.195, -1.345]
                  }
    """

    # define input string
    data = data
    length = len(data)
    # define empty array
    data_X = np.zeros((length, 2*windows+1, 5))
    for i in range(length):
        x = data[i].split()
        # 编码氨基酸理化属性
        j = 0
        for AA in x[2]:
            for index, value in enumerate(letterDict[AA]):
                data_X[i][j][index] = value
            j = j + 1

    return data_X


# 说明： 氨基酸理化信息编码
# 输入： data
# 输出： data_X
# 来源： 2018年论文“Capsule network for protein post-translational modification site prediction”所附代码中获取到的
def Phy_Chem_Inf_1(data, windows=16):
    letterDict = {}  # hydrophobicty, hydrophilicity, side-chain mass, pK1 (alpha-COOH), pK2 (NH3), PI, Average volume of buried residue, Molecular weight, Side chain volume, Mean polarity
    letterDict["A"] = [0.62, -0.5, 15, 2.35, 9.87, 6.11, 91.5, 89.09, 27.5, -0.06]
    letterDict["C"] = [0.2900, -1.0000, 47.0000, 1.7100, 10.7800, 5.0200, 117.7, 121.15, 44.6, 1.36]
    letterDict["D"] = [-0.9000, 3.0000, 59.0000, 1.8800, 9.6000, 2.9800, 124.5, 133.1, 40, -0.8]
    letterDict["E"] = [-0.7400, 3.0000, 73.0000, 2.1900, 9.6700, 3.0800, 155.1, 147.13, 62, -0.77]
    letterDict["F"] = [1.1900, -2.5000, 91.0000, 2.5800, 9.2400, 5.9100, 203.4, 165.19, 115.5, 1.27]
    letterDict["G"] = [0.4800, 0, 1.0000, 2.3400, 9.6000, 6.0600, 66.4, 75.07, 0, -0.41]
    letterDict["H"] = [-0.4000, -0.5000, 82.0000, 1.7800, 8.9700, 7.6400, 167.3, 155.16, 79, 0.49]
    letterDict["I"] = [1.3800, -1.8000, 57.0000, 2.3200, 9.7600, 6.0400, 168.8, 131.17, 93.5, 1.31]
    letterDict["K"] = [-1.5000, 3.0000, 73.0000, 2.2000, 8.9000, 9.4700, 171.3, 146.19, 100, -1.18]
    letterDict["L"] = [1.0600, -1.8000, 57.0000, 2.3600, 9.6000, 6.0400, 167.9, 131.17, 93.5, 1.21]
    letterDict["M"] = [0.6400, -1.3000, 75.0000, 2.2800, 9.2100, 5.7400, 170.8, 149.21, 94.1, 1.27]
    letterDict["N"] = [-0.7800, 0.2000, 58.0000, 2.1800, 9.0900, 10.7600, 135.2, 132.12, 58.7, -0.48]
    letterDict["P"] = [0.1200, 0, 42.0000, 1.9900, 10.6000, 6.3000, 129.3, 115.13, 41.9, 0]
    letterDict["Q"] = [-0.8500, 0.2000, 72.0000, 2.1700, 9.1300, 5.6500, 161.1, 146.15, 80.7, -0.73]
    letterDict["R"] = [-2.5300, 3.0000, 101.0000, 2.1800, 9.0900, 10.7600, 202, 174.2, 105, -0.84]
    letterDict["S"] = [-0.1800, 0.3000, 31.0000, 2.2100, 9.1500, 5.6800, 99.1, 105.09, 29.3, -0.5]
    letterDict["T"] = [-0.0500, -0.4000, 45.0000, 2.1500, 9.1200, 5.6000, 122.1, 119.12, 51.3, -0.27]
    letterDict["V"] = [1.0800, -1.5000, 43.0000, 2.2900, 9.7400, 6.0200, 141.7, 117.15, 71.5, 1.09]
    letterDict["W"] = [0.8100, -3.4000, 130.0000, 2.3800, 9.3900, 5.8800, 237.6, 204.24, 145.5, 0.88]
    letterDict["Y"] = [0.2600, -2.3000, 107.0000, 2.2000, 9.1100, 5.6300, 203.6, 181.19, 117.3, 0.33]
    letterDict["-"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["B"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["J"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["O"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["U"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["X"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["Z"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # define input string
    data = data
    length = len(data)
    # define empty array
    data_X = np.zeros((length, 2*windows+1, 10))
    for i in range(length):
        x = data[i].split()
        # 编码氨基酸理化属性
        j = 0
        for AA in x[2]:
            for index, value in enumerate(letterDict[AA]):
                data_X[i][j][index] = value
            j = j + 1

    return data_X


# 说明： 氨基酸理化信息编码
# 输入： data
# 输出： data_X
# 来源： 2018年论文“Capsule network for protein post-translational modification site prediction”，237种属性总结为5类，再加一维表示是否为空AA
def Phy_Chem_Inf_2(data, windows=16):
    letterDict = {}
    letterDict["A"] = [0.008, 0.134, -0.475, -0.039, 0.181, 0]
    letterDict["C"] = [-0.132, 0.174, 0.070, 0.565, -0.374, 0]
    letterDict["D"] = [0.303, -0.057, -0.014, 0.225, 0.156, 0]
    letterDict["E"] = [0.221, -0.280, -0.315, 0.157, 0.303, 0]
    letterDict["F"] = [-0.329, -0.023, 0.072, -0.002, 0.208, 0]
    letterDict["G"] = [0.218, 0.562, -0.024, 0.018, 0.106, 0]
    letterDict["H"] = [0.023, -0.177, 0.041, 0.280, -0.021, 0]
    letterDict["I"] = [-0.353, 0.071, -0.088, -0.195, -0.107, 0]
    letterDict["K"] = [0.243, -0.339, -0.044, -0.325, -0.027, 0]
    letterDict["L"] = [-0.267, 0.018, -0.265, -0.274, 0.206, 0]
    letterDict["M"] = [-0.239, -0.141, -0.155, 0.321, 0.077, 0]
    letterDict["N"] = [0.255, 0.038, 0.117, 0.118, -0.055, 0]
    letterDict["P"] = [0.173, 0.286, 0.407, -0.215, 0.384, 0]
    letterDict["Q"] = [0.149, -0.184, -0.030, 0.035, -0.112, 0]
    letterDict["R"] = [0.171, -0.361, 0.107, -0.258, -0.364, 0]
    letterDict["S"] = [0.199, 0.238, -0.015, -0.068, -0.196, 0]
    letterDict["T"] = [0.068, 0.147, -0.015, -0.132, -0.274, 0]
    letterDict["V"] = [-0.274, 0.136, -0.187, -0.196, -0.299, 0]
    letterDict["W"] = [-0.296, -0.186, 0.389, 0.083, 0.297, 0]
    letterDict["Y"] = [-0.141, -0.057, 0.425, -0.096, -0.091, 0]
    letterDict["-"] = [0.0, 0.0, 0.0, 0.0, 0.0, 1]
    letterDict["B"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0]
    letterDict["J"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0]
    letterDict["O"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0]
    letterDict["U"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0]
    letterDict["X"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0]
    letterDict["Z"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0]

    # define input string
    data = data
    length = len(data)
    # define empty array
    data_X = np.zeros((length, 2*windows+1, 6))
    for i in range(length):
        x = data[i].split()
        # 编码氨基酸理化属性
        j = 0
        for AA in x[2]:
            for index, value in enumerate(letterDict[AA]):
                data_X[i][j][index] = value
            j = j + 1

    return data_X



# 说明： 蛋白质结构信息编码
# 输入： data
# 输出： data_X
def Structure_Inf(data, windows=16):

    # define input string
    data = data
    length = len(data)
    # define empty array
    data_X = np.zeros((length, 2*windows+1, 8))
    for i in range(length):
        x = data[i].split()
        # print(x)
        # print(x[3])
        # 编码蛋白质结构信息
        f_r = open("./dataset/Generic/Structure_information/%s.i1" % x[0], "r", encoding='utf-8')
        lines = f_r.readlines()
        List = []
        for line in lines:
            z = line.split()
            if z[0] != '#':
                List.append(line)
        f_r.close()
        # print(List)
        # print(List[int(x[3])])
        # 检查List和data中赖氨酸位置标识是否相同
        k = List[int(x[3])].split()
        if int(k[0]) != int(x[3]) + 1:
            exit()
        j = 0
        offset = 0
        for AA in x[2]:
            if AA != '-':
                value = List[int(x[3]) - windows + offset].split()
                data_X[i][j][0] = value[4]
                data_X[i][j][1] = value[5]
                data_X[i][j][2] = value[6]
                data_X[i][j][3] = value[7]
                data_X[i][j][4] = value[8]
                data_X[i][j][5] = value[14]
                data_X[i][j][6] = value[13]
                data_X[i][j][7] = value[12]
            else:
                data_X[i][j][0] = 0.0
                data_X[i][j][1] = 0.0
                data_X[i][j][2] = 0.0
                data_X[i][j][3] = 0.0
                data_X[i][j][4] = 0.0
                data_X[i][j][5] = 0.0
                data_X[i][j][6] = 0.0
                data_X[i][j][7] = 0.0
            j = j + 1
            offset = offset + 1
    # print(data_X)
    return data_X

if __name__ == '__main__':
    # Structure_Inf(['P13804 0 ADLFKVVPEMTEILKKK---------------- 332\n'
    #                'Q8IZQ5 0 LLRPDGSSAELWTGIKKGPPRKLKFPEPQEVVE 98\n'], windows=16)
    one_hot(['Q8IZQ5 0 LLRPDGSSAELWTGIKKGPPRKLKFPEPQEVVE 98\n'], windows=16)


