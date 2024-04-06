import sys

def main(index):
    # Your code here
    print(f"Running job with index {index}")

if __name__ == "__main__":
    index = int(sys.argv[1])
    main(index)