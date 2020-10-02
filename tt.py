inr =  int(input())
lis =  []
for i in range(inr):
    s1 = input()
    s1 = s1.split(' ')
    if len(s1) == 3:
        s1,s2,s3= s1
    else:
        s1 = s1[0]
    if s1 == '1':
        if len(lis) >= int(s2) :
            lis.insert(int(s2),s3)
    elif s1 == '2':
        if len(lis)- 1>= int(s2):
            del lis[int(s2)]
    else:
        print(lis)
