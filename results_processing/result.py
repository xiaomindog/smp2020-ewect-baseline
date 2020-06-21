import json

label_list = ['neural', 'sad', 'happy', 'fear', 'angry', 'surprise']

id_list = []

with open("../SMP/eval/all_eval.json", 'r', encoding='utf-8')  as f:
    for line in f.readlines():
        datas = json.loads(line)
        for data in datas:
        #print(data)
            id_list.append(data['id'])

virus_result = []
usual_result = []
with open('smp2020ewect-bert-wwm-nl2sql.txt', 'r', encoding='utf-8') as f:
    for lid, line in enumerate(f.readlines()):
        line = line.strip()
        virus = {}
        usual = {}
        ss = id_list[lid].split("_")
        if ss[0] == "virus":
            virus['id'] = int(ss[1])
            virus['label'] = label_list[int(line)]
            virus_result.append(virus)
        elif ss[0] == 'usual':
            usual['id'] = int(ss[1])
            usual['label'] = label_list[int(line)]
            usual_result.append(usual)

with open('./virus-nl2sql.json','w',encoding='utf-8') as f:
    json.dump(virus_result,f)
with open('./usual-nl2sql.json','w',encoding='utf-8') as f:
    json.dump(usual_result,f)