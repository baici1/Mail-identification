import requests

url = "http://127.0.0.1:5000/result"
data = {"text":"这个是不是垃圾邮件"}
res = requests.post(url=url,data=data).text
print(res)

