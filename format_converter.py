import pickle


with open('trainingLabel.pkl', 'rb') as dataf:
    dataFile = pickle.load(dataf)
print(dataFile[0])

data = []
for line in dataFile:
    # is_immoral
    if (line[0] == True):
        is_immoral = 1
    elif (line[0] == False):
        is_immoral = 0
    else:
        raise
    
    # types
    types = []
    for type in line[1]:
        if (type == 'IMMORAL_NONE'):
            types.append(0)
        elif (type == 'DISCRIMINATION'):
            types.append(1)
        elif (type == 'HATE'):
            types.append(2)
        elif (type == 'CENSURE'):
            types.append(3)
        elif (type == 'VIOLENCE'):
            types.append(4)
        elif (type == 'CRIME'):
            types.append(5)
        elif (type == 'SEXUAL'):
            types.append(6)
        elif (type == 'ABUSE'):
            types.append(7)
        elif (type == 'IMMORAL_NONE'):
            types.append(8)
        else:
            raise
    
    # intensity
    intensity = line[2]

    # 합치기
    data.append((is_immoral, types, intensity))

with open('final_trainingLabel.pkl', 'wb') as fw:
    pickle.dump(data, fw)
