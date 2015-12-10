import sys

def grep(filename, needle):
    with open(filename) as f_in:
        matches = ((i, line.find(needle), line) for i, line in enumerate(f_in))
    	return [match for match in matches if match[0] != -1]

def main(filename, needle):
	matches = grep(filename, needle)
	print matches

if __name__=='__main__':
    filename = sys.argv[1]
    needle = sys.argv[2]
    main(filename, needle)