import math
import sys


def compute_fun(filename):
    data=[]
    print(f"Processing file: {filename}")

    #open the file
    with open(filename, 'r') as file:
        for line in file:
            numbers=list(map(float,line.split()))
            data.append(numbers)
    print(data)

    #if file is empty
    if not data or not data[0]:
        print("No data found in file.")
        return

    #number of column
    num_columns = len(data[0])
    print(num_columns)

    for col in range(num_columns):
        #get the column value
        column_values = [row[col] for row in data]

        # Compute the mean
        mean = sum(column_values) / len(column_values)

        # Compute the standard deviation
        variance = sum((x - mean) ** 2 for x in column_values) / (len(column_values) - 1)
        std_dev = math.sqrt(variance)

        print(f"Column {col + 1}: mean = {mean:.4f}, std = {std_dev:.4f}")
def main():
    # Check if a file path was provided as a command line argument
    if len(sys.argv) != 2:
        print("Usage: python file_stats.py <filename>")
        sys.exit(1)

    # Get the file path from the command line argument
    filename = sys.argv[1]

    # Compute stats for the given file
    compute_fun(filename)

if __name__ == "__main__":
    main()

