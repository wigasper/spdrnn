import re
import os
from pathlib import Path
from random import randint

NUM_SAMPS_PER_SEC = 8
NUM_OBS_PER_SAMP = 400

def in_trigger_interval(time, trigger_intervals):
    for trigger_int in trigger_intervals:
        if time > trigger_int[0] and time <= trigger_int[1]:
            return True

    return False

def min_max_normalize(samples):
    data_points = [float(it[1]) for it in samples]
    minimum = min(data_points)
    maximum = max(data_points)

    data_out = []
    
    denom = maximum - minimum

    for sample in samples:
        x_prime = (float(sample[1]) - minimum) / denom

        data_out.append([sample[0], x_prime])

    return data_out

def load_data(file_path, start_time, stop_time):
    data = []

    with open(file_path, "r") as handle:
        current_time = 0
        current_time_slice = []
        current_time_points = []

        for line in handle:
            line = line.split()
            
            line_time = line[1].split(".")[0]
            if float(line_time) >= start_time and float(line_time) < stop_time:
                if line_time != current_time:
                    current_time = line_time

                    if len(current_time_slice) > 0:
                        pctile_idx = int(len(current_time_slice) / NUM_SAMPS_PER_SEC)

                        for mult in range(NUM_SAMPS_PER_SEC):
                            data.append([current_time_points[mult * pctile_idx], 
                                current_time_slice[mult * pctile_idx]])

                    current_time_slice = [line[0]]
                    current_time_points = [line[1]]
                else:
                    current_time_slice.append(line[0])
                    current_time_points.append(line[1])
    
    return data    

def get_dir_list(directory):
    absolute_path = Path(directory).resolve()
    files = os.listdir(directory)
    absolute_fps = [os.path.join(absolute_path, f) for f in files]

    return [fp for fp in absolute_fps if os.path.isdir(fp)]

def process_dir(dir_path):
    uid = dir_path.split("/")[-1]
    trigger_intervals = []

    with open(f"{dir_path}/Triggers.txt", "r") as handle:
        for line in handle:
            if re.search("^CLIP-\d.*", line):
                line = line.split()
                trigger_intervals.append([int(line[1]), int(line[2])])
    
    temp_trigger_times = []

    for interval in trigger_intervals:
        temp_trigger_times.append(interval[0])
        temp_trigger_times.append(interval[1])

    start_time = min(temp_trigger_times)
    stop_time = max(temp_trigger_times)

    resp_data = load_data(f"{dir_path}/BitalinoBR.txt", start_time, stop_time)
    ecg_data = load_data(f"{dir_path}/BitalinoECG.txt", start_time, stop_time)
    skin_data = load_data(f"{dir_path}/BitalinoGSR.txt", start_time, stop_time)
   
    resp_norm = min_max_normalize(resp_data)
    ecg_norm = min_max_normalize(ecg_data)
    skin_norm = min_max_normalize(skin_data)
    
    partition = int(len(resp_norm) * 0.75)
    
    n_test_samples = int(2 * (len(resp_norm) - partition) / NUM_OBS_PER_SAMP)

    for idx in range(n_test_samples):
        start_idx = randint(partition, len(resp_norm) - NUM_OBS_PER_SAMP)
        
        while in_trigger_interval(float(resp_norm[start_idx][0]), trigger_intervals):
            start_idx = randint(partition, len(resp_norm) - NUM_OBS_PER_SAMP)

        end_idx = start_idx + NUM_OBS_PER_SAMP
        with open(f"../data/test/{uid}_{idx}", "w") as out:
            for i in range(start_idx, end_idx):
                response = 0
                if in_trigger_interval(float(resp_norm[i][0]), trigger_intervals):
                    response = 1
                out.write(f"{resp_norm[i][0]},{resp_norm[i][1]},{ecg_norm[i][1]},")
                out.write(f"{skin_norm[i][1]},{response}\n")
                #resp_norm.pop(i)
                #ecg_norm.pop(i)
                #skin_norm.pop(i)
       
    n_training_samples = int(5 * partition / NUM_OBS_PER_SAMP)

    for idx in range(n_training_samples):
        start_idx = randint(0, partition - NUM_OBS_PER_SAMP)
        
        while in_trigger_interval(float(resp_norm[start_idx][0]), trigger_intervals):
            start_idx = randint(0, partition - NUM_OBS_PER_SAMP)

        end_idx = start_idx + NUM_OBS_PER_SAMP
        with open(f"../data/train/{uid}_{idx}", "w") as out:
            for i in range(start_idx, end_idx):
                response = 0
                if in_trigger_interval(float(resp_norm[i][0]), trigger_intervals):
                    response = 1
                out.write(f"{resp_norm[i][0]},{resp_norm[i][1]},{ecg_norm[i][1]},")
                out.write(f"{skin_norm[i][1]},{response}\n")
    

if __name__ == "__main__":
    dirs = get_dir_list(".")

    for this_dir in dirs:
        process_dir(this_dir)
#    with open("all_data", "w") as out:
#        for idx, resp_samp in enumerate(resp_norm):
#            response = 0
#            if in_trigger_interval(float(resp_samp[0]), trigger_intervals):
#                response = 1
#            out.write(f"{idx},{resp_samp[1]},{ecg_norm[idx][1]},")
#            out.write(f"{skin_norm[idx][1]},{response}\n")
#    print(f"ecg len: {len(ecg_data)}")
#    print(f"resp len: {len(resp_data)}")
#    print(f"skin len: {len(skin_data)}")
#    print(f"ecgn len: {len(ecg_norm)}")
#    print(f"respn len: {len(resp_norm)}")
#    print(f"skinn len: {len(skin_norm)}")
