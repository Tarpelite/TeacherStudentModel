import csv

in_path = "share/"
out_path = "small_data/"


def sample_data(sample_rate):
    train_path =  "train.csv"
    paths  = ["train.csv", "dev.csv", "test.csv"]
    for file_path in paths:
        with open(in_path + file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            lines = []
            for line in reader:
                lines.append(line)
        sample_len = int(len(lines)*sample_rate)
        out_lines = lines[:sample_len]
        with open(out_path + file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter = "\t")
            for line in out_lines:
                writer.writerow(line)
        if file_path == "dev.csv":
            with open(out_path + "aus_dev.label.csv", "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                out_lines = [[x[1]] for x in out_lines]
                for line in out_lines:
                    writer.writerow(line)

def create_testset(sample_rate, dev_path):
    '''
        to split dev data into test while keeping the distribution rate
    '''
    train_path = 

        

if __name__ == "__main__":
    sample_rate = 0.1
    sample_data(sample_rate)
