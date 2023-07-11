import json
def create_json(keys, values):
    data = {}
    for i in range(len(keys)):
        data[keys[i]] = values[i]
    json_data = json.dumps(data) #, separators=(',', ':'))
    return json_data

keys = ["label","position","confidence", "left shoulder", "left elbow", 
       "left wrist", "right shoulder", "right elbow", "right wrist", "object label", "object confidence"]
values = ["person","Left",0.41, [42,83],[46,100],[41,110],[21,82],[12,71],[5, 64],"bicycle",True,2867,0.84,[342, 152, 395, 294]]

json_string = create_json(keys, values)
print(json_string)