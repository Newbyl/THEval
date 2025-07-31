import os
import argparse

def find_mp4_files(folder):
    for root, _, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith('.mp4'):
                yield os.path.abspath(os.path.join(root, fname))

def main():
    parser = argparse.ArgumentParser(description="List all .mp4 files in given folder(s) and their subfolders")
    parser.add_argument('folders', nargs='+', help='Path(s) to the folder(s) to search')
    parser.add_argument('-o', '--output',
                        help='Path to output txt file',
                        default='mp4_paths.txt')
    args = parser.parse_args()

    with open(args.output, 'w') as out_f:
        for folder in args.folders:
            for path in find_mp4_files(folder):
                out_f.write(path + '\n')

    print(f"All MP4 paths written to {args.output}")

if __name__ == "__main__":
    main()