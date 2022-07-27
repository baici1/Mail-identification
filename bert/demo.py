import requests

url = "http://127.0.0.1:5001/result"
data = {"text":"红楼梦写到大观园试才题对额时有一个情节，为元妃(贾元春)省亲修建的大观园竣工后，众人给园中桥上亭子的匾额题名"}
res = requests.post(url=url,data=data).text
print(res)

