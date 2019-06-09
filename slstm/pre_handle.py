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

big2_length = 0
big4_length = 0
big8_length = 0
big16_length = 0
big32_length = 0

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
                big2_length = big2_length+1
                d2.append({'bid': bid, 'bout': bout, 'ret': ret, 'states': copy(states), 'interval': copy(times)})

            if step == 4:
                big4_length = big4_length + 1
                d4.append({'bid': bid, 'bout': bout, 'ret': ret, 'states': copy(states), 'interval': copy(times)})

            if step == 8:
                big8_length = big8_length + 1
                d8.append({'bid': bid, 'bout': bout, 'ret': ret, 'states': copy(states), 'interval': copy(times)})

            if step == 16:
                big16_length = big16_length + 1
                d16.append({'bid': bid, 'bout': bout, 'ret': ret, 'states': copy(states), 'interval': copy(times)})

            if step == 32:
                big32_length = big32_length + 1
                d32.append({'bid': bid, 'bout': bout, 'ret': ret, 'states': copy(states), 'interval': copy(times)})

small2_length = 0
small4_length = 0
small8_length = 0
small16_length = 0
small32_length = 0
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
                small2_length = small2_length + 1
                if small2_length >= big2_length:
                    continue
                d2.append({'bid': bid, 'bout': bout, 'ret': ret, 'states': copy(states), 'interval': copy(times)})

            if step == 4:
                small4_length = small4_length + 1
                if small4_length >= big4_length:
                    continue
                d4.append({'bid': bid, 'bout': bout, 'ret': ret, 'states': copy(states), 'interval': copy(times)})

            if step == 8:
                small8_length = small8_length + 1
                if small8_length >= big8_length:
                    continue
                d8.append({'bid': bid, 'bout': bout, 'ret': ret, 'states': copy(states), 'interval': copy(times)})

            if step == 16:
                small16_length = small16_length + 1
                if small16_length >= big16_length:
                    continue
                d16.append({'bid': bid, 'bout': bout, 'ret': ret, 'states': copy(states), 'interval': copy(times)})

            if step == 32:
                small32_length = small32_length + 1
                if small32_length >= big32_length:
                    continue
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

print("big2_length "+str(big2_length))
print("big4_length "+str(big4_length))
print("big8_length "+str(big8_length))
print("big16_length "+str(big16_length))
print("big32_length "+str(big32_length))
print("small2_length "+str(small2_length))
print("small4_length "+str(small4_length))
print("small8_length "+str(small8_length))
print("small16_length "+str(small16_length))
print("small32_length "+str(small32_length))