import csv
import json
import random

inteval_times = []
d2 = []
d4 = []
d8 = []
d16 = []
d32 = []

c0 = 0
c1 = 0
c2 = 0
c3 = 0
c4 = 0
c5 = 0

def copy(items):
    array = []
    for item in items:
        array.append(item)
    return array


with open('data/fixed34tempo.csv') as f:
    f_csv = csv.reader(f)
    skip_one = 1
    for row in f_csv:
        if skip_one == 1:
            skip_one = 0
            continue

        bid = row[0]
        bout = 1
        ret = random.random()

        b_stats = row[1][2:len(row[1]) - 2].split("), (")

        if len(b_stats) == 1:
            continue

        pre_time = 0
        step = 0
        states = []
        times = []
        for b in b_stats:
            index = b.find("', '")
            state = b[1:index]
            time = int(b[index + 4:len(b) - 1])

            if time > 32:
                continue

            step = step + 1
            if step > 32:
                continue

            inteval_time = time - pre_time
            pre_time = time
            inteval_time_flag = 0

            c0 = c0 + 1
            if inteval_time <= 0:
                c1 = c1 + 1
                inteval_time_flag = 0
            elif inteval_time <= 2:
                c2 = c2 + 1
                inteval_time_flag = 1
            elif inteval_time <= 4:
                c3 = c3 + 1
                inteval_time_flag = 2
            elif inteval_time <= 8:
                c4 = c4 + 1
                inteval_time_flag = 3
            else:
                c5 = c5 + 1
                inteval_time_flag = 4

            states.append(state)
            times.append(inteval_time_flag)

            if step == 2:
                d2.append({'bid': bid, 'bout': bout, 'ret': ret, 'states': copy(states), 'interval': copy(times)})

            if step == 4:
                d4.append({'bid': bid, 'bout': bout, 'ret': ret, 'states': copy(states), 'interval': copy(times)})

            if step == 8:
                d8.append({'bid': bid, 'bout': bout, 'ret': ret, 'states': copy(states), 'interval': copy(times)})

            if step == 16:
                d16.append({'bid': bid, 'bout': bout, 'ret': ret, 'states': copy(states), 'interval': copy(times)})

            if step == 32:
                d32.append({'bid': bid, 'bout': bout, 'ret': ret, 'states': copy(states), 'interval': copy(times)})

with open('data/fixed35tempo.csv') as f:
    f_csv = csv.reader(f)
    skip_one = 1
    for row in f_csv:
        if skip_one == 1:
            skip_one = 0
            continue

        bid = row[0]
        bout = 0
        ret = random.random()

        b_stats = row[1][2:len(row[1]) - 2].split("), (")

        if len(b_stats) == 1:
            continue

        pre_time = 0
        step = 0
        states = []
        times = []
        for b in b_stats:
            index = b.find("', '")
            state = b[1:index]
            time = int(b[index + 4:len(b) - 1])

            if time > 32:
                continue

            step = step + 1
            if step > 32:
                continue

            inteval_time = time - pre_time
            pre_time = time
            inteval_time_flag = 0

            c0 = c0 + 1
            if inteval_time <= 0:
                c1 = c1 + 1
                inteval_time_flag = 0
            elif inteval_time <= 2:
                c2 = c2 + 1
                inteval_time_flag = 1
            elif inteval_time <= 4:
                c3 = c3 + 1
                inteval_time_flag = 2
            elif inteval_time <= 8:
                c4 = c4 + 1
                inteval_time_flag = 3
            else:
                c5 = c5 + 1
                inteval_time_flag = 4

            states.append(state)
            times.append(inteval_time_flag)

            if step == 2:
                d2.append({'bid': bid, 'bout': bout, 'ret': ret, 'states': copy(states), 'interval': copy(times)})

            if step == 4:
                d4.append({'bid': bid, 'bout': bout, 'ret': ret, 'states': copy(states), 'interval': copy(times)})

            if step == 8:
                d8.append({'bid': bid, 'bout': bout, 'ret': ret, 'states': copy(states), 'interval': copy(times)})

            if step == 16:
                d16.append({'bid': bid, 'bout': bout, 'ret': ret, 'states': copy(states), 'interval': copy(times)})

            if step == 32:
                d32.append({'bid': bid, 'bout': bout, 'ret': ret, 'states': copy(states), 'interval': copy(times)})


with open("data/d2.json", "w") as f:
    json.dump(d2, f)

with open("data/d4.json", "w") as f:
    json.dump(d4, f)

with open("data/d8.json", "w") as f:
    json.dump(d8, f)

with open("data/d16.json", "w") as f:
    json.dump(d16, f)

with open("data/d32.json", "w") as f:
    json.dump(d32, f)

# print(c0)
# print(c1)
# print(c2)
# print(c3)
# print(c4)
# print(c5)