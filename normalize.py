

idx = 0;
for line in open('./submit.txt', 'r'):
    if idx == 0:
        print(line, end='');
    else:
        probs = [float(v) for v in line.strip('\n').split(',')];
        probs = probs[1:];
        m = min(probs);
        print(m)
        probs = [v-m for v in probs];
        print(str(idx) + ',' + ','.join([str(v) for v in probs]));
    idx += 1;
