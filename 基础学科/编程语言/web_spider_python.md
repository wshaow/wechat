# 小甲虫

## urllib模块

```python
import urllib.request
import json

if __name__ == '__main__':
    # url = 'http://fanyi.youdao.com/translate_o?smartresult=dict&smartresult=rule'
    # data = {'i':'你好',
    #         'from':'AUTO',
    #         'to':'AUTO',
    #         'smartresult':'dict',
    #         'client':'fanyideskweb',
    #         'salt':'15836536687512',
    #         'sign':'ad5ac2d1841bd1944f60d857fdf55e6c',
    #         'ts':'1583653668751',
    #         'bv':'a9c3483a52d7863608142cc3f302a0ba',
    #         'doctype':'json',
    #         'version':'2.1',
    #         'keyfrom':'fanyi.web',
    #         'action':'FY_BY_CLICKBUTTION'}
    # data = urllib.parse.urlencode(data).encode('utf-8')
    import urllib.parse
    url = 'http://fanyi.youdao.com/translate?&doctype=json&type=AUTO&i=hello'
    response = urllib.request.urlopen(url)
    html = response.read().decode('utf-8')
    print(html)
    print(type(html))
    target = json.loads(html)
    print(type(target))
    print(target['translateResult'][0][0]['tgt'])

```

上面把翻译内容换成中文会有编码问题

编码问题有点没有弄清楚

使用代理解决访问频率过快被屏蔽的问题

![image-20200308203155610](web_spider_python.assets/image-20200308203155610.png)

这个课程有点老，放弃

# 慕课北理工

![image-20200308210009562](web_spider_python.assets/image-20200308210009562.png)



