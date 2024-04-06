import sys

def main(job, index):
    print("Running job with job {}".format(job))
    print("Running job with index {}".format(index))

if __name__ == "__main__":
    index = int(sys.argv[2])
    job = int(sys.argv[1])
    main(job, index)