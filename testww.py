string =  input()
N,M = string.split()
N = int(N)
M = int(M)
infor = []
for i in range(N):
    string = input()
    infor.append(string.split())

dictm = {}
for i in range(len(infor)):
    key =  infor[i][0]
    if key in dictm:
        dictm[key].append(infor[i][1])
    else:
        dictm[key] = [infor[i][1]]

    key =  infor[i][1]
    if infor[i][1] in dictm:
        dictm[key].append(infor[i][0])
    else:
        dictm[key] = [infor[i][1]]

def find(data,key):
    listr = data[key]
    listrmm = {}
    while(len(listr) != 0):
        keys =  listr.pop()
        if keys in listrmm:
            pass
        else:
            listrmm[keys] = None
            listr = data[keys] + listr
    if key in listrmm:
        mm = []
    else:
        mm = [key]
    for key in listrmm:
        mm.append(key)
    return mm
rr = []
while(len(dictm)!= 0):
    for key in dictm:
        key1 = key
        break
    listrmm = find(dictm,key1)
    for key in listrmm:
        del dictm[key]
    rr.append(listrmm)
value = []

for ii in range(len(rr)):
    for j in range(len(rr[ii])):
        rr[ii][j] = int(rr[ii][j])
    rr[ii] = sorted(rr[ii])
    value.append(rr[ii][0])

while(len(value )!=0):
    data = min(value)
    index = value.index(data)
    print(data)
    print(rr[index])
    del value[index]




