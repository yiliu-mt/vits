import argparse
import os
from multiprocessing import Process


def do_resample(resample, filelist, input_dir, output_dir):
    for file in filelist:
        input_file = os.path.join(input_dir, file)
        output_file = os.path.join(output_dir, file)
        os.system("sox {} -r {} {}".format(input_file, resample, output_file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', type=int, default=8, help='number of jobs')
    parser.add_argument("-r", type=int, required=True, help="The resample rate")
    parser.add_argument("input_dir", help="The input directory")
    parser.add_argument("output_dir", help="The output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # scan wave files
    wav_list = []
    for filename in os.listdir(args.input_dir):
        if filename.endswith(".wav"):
            wav_list.append(filename)

    # do work
    split_list = [[] for _ in range(args.j)]
    for i, wav in enumerate(wav_list):
        split_list[i % args.j].append(wav)

    processes = [Process(
        target=do_resample, args=(args.r, split_list[i], args.input_dir, args.output_dir)
    ) for i in range(args.j)]
    for proc in processes:
        proc.daemon = True
        proc.start()
    for proc in processes:
        proc.join()


if __name__ == '__main__':
    main()