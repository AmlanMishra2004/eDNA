import collections
import re

def print_max_accuracy(file_name):
    max_accuracy = 0.0
    lines_buffer = collections.deque(maxlen=200)
    result_lines = []

    with open(file_name, 'r') as file:
        for line in file:
            lines_buffer.append(line)

            try:
                match = re.search(r"Val acc at epoch \d+: (\d+\.\d+)", line)
                if match:
                    accuracy = float(match.group(1))
                    if accuracy > max_accuracy:
                        max_accuracy = accuracy
                        result_lines = list(lines_buffer)
            except ValueError:
                pass
    print(f"Best result: ")
    for line in result_lines:
        print(line, end='')

print_max_accuracy('out.1806442.log')

