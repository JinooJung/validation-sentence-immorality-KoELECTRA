import json
import os
import pickle

def dot_adder(text : str) :
    if (len(text)>0):
        if ((text[-1] == '.') or (text[-1] == '?') or (text[-1] == '!') or (text[-1] == '"')):
            text += " "
        else :
            text += ". "
    else :
        text += ""
    return text


""" parser init """
directory_path = "../Data/Training"
# must .pkl added"
saveFileName_input = "trainingData.pkl"
saveFileName_result = "trainingLabel.pkl"    
"""             """

pklWriter_input = open(saveFileName_input,"wb")
pklWriter_result = open(saveFileName_result,"wb")

file_names = os.listdir(directory_path)
files = [directory_path + "/" + file_name for file_name in file_names]


dataSet_input = []
dataSet_result = []
for file in files:
    print(file)

    # 파일 하나 load 해오기
    with open(file, 'r', encoding='utf-8') as f:
        load_data = json.load(f)
    
    print("\t file load complete")

    # for문 돌면서 하나씩 저장
    for sentences in load_data:
        for sentence in sentences['sentences']:
            # data
            dataSet_input.append(sentence['origin_text'])

            # label
            dataSet_result.append((sentence['is_immoral'],sentence['types'],sentence['intensity']))
    f.close()

# 파일 하나 정리한거 저장
pickle.dump(dataSet_input, pklWriter_input)
pickle.dump(dataSet_result, pklWriter_result)
pklWriter_input.close()
pklWriter_result.close()


