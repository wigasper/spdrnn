#!/usr/bin/env python3

import math

from random import randrange

if __name__ == "__main__":
    num_samples = 300
    
    for samp in range(num_samples):
        x = []
        y = []
        start_num = randrange(1, 8)
        for num in range(start_num, start_num + 200):
            num = num * 0.1
            #multiplicand = randrange(95, 105, 1) / 100
            #multiplicand = 1
            offset = randrange(98, 102, 1) / 100
            x.append(math.sin(num) + offset)
            #x.append(math.sin(num) + math.sin(num) * multiplicand)

            if (math.pi / 2) < (num % (2 * math.pi)) <= (3 * math.pi / 2):
                y.append(0)
            else:
                y.append(1)

        with open(f"train/{samp}", "w") as out:
            for idx, x_val in enumerate(x):
                out.write(f"{x_val},{y[idx]}\n")
    
    num_samples = 75
    
    for samp in range(num_samples):
        x = []
        y = []
        start_num = randrange(1, 8)
        for num in range(start_num, start_num + 200):
            num = num * 0.1
            #multiplicand = randrange(95, 105, 1) / 100
            #multiplicand = 1
            offset = randrange(98, 102, 1) / 100
            x.append(math.sin(num) + offset)
            #x.append(math.sin(num) + math.sin(num) * multiplicand)

            if (math.pi / 2) < (num % (2 * math.pi)) <= (3 * math.pi / 2):
                y.append(0)
            else:
                y.append(1)

        with open(f"test/{samp}", "w") as out:
            for idx, x_val in enumerate(x):
                out.write(f"{x_val},{y[idx]}\n")
       
