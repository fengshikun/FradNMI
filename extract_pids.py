# ps -ef | grep train > kill.temp

pids = []
with open('kill.temp', 'r') as kr:
    for line in kr.readlines():
        aa = line.strip().split()[1]
        pids.append(aa)
print('kill  -9 ' +  ' '.join(pids))