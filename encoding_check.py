import chardet

with open('data.csv', 'rb') as f:
    result = chardet.detect(f.read())
    print(result)
