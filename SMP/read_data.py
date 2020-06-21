import json
#
# s = set()
#
papers = []
with open('./train/virus_train.json', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        dic = json.loads(line)
        dic['id'] = "virus_"+str(dic['id'])
        papers.append(dic)
        print(dic)
#
with open('./train/usual_train.json', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        dic = json.loads(line)
        dic['id'] = "usual_" + str(dic['id'])
        papers.append(dic)
#
with open('./train/all_train.json', 'w', encoding='utf-8') as f:

    json.dump(papers,f)


with open('./train/all_train.json', 'r', encoding='utf-8') as f:
    for line in f.readlines():
         dic = json.loads(line)
         print(dic[3999])