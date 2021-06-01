import argparse
import os
import re

from Bio import SeqIO
import numpy as np


def get_normal_dist(mean, std):
    return np.random.normal(loc=mean, scale=std)


def main(args):
    reference_path = os.path.abspath(args.ref)
    out_path = os.path.abspath(args.out)
    assert re.findall('.*\.([A-Za-z]*)', out_path)[-1] in ('fastq', 'fq'), \
            "Output path should be in the FASTQ format"
    depth = args.depth
    length_mean = args.length_mean
    length_std = args.length_std if args.length_std is not None else length_mean * 0.075

    accuracy = 1.0
    reference = next(SeqIO.parse(reference_path, 'fasta'))
    reference_rc = reference.reverse_complement()
    num_reads = len(reference) // length_mean * depth
    step_mean = (len(reference) - length_mean) / (num_reads - 1)
    step_mean = round(step_mean / 100) * 100
    step_std = step_mean * 0.1
    start, stop = 0, len(reference)
    reads_list = []

    idx = 0
    position = 0
    while position < stop:
        length = int(get_normal_dist(length_mean, length_std))
        if position + length < len(reference):
            read = reference[position:position+length]
        else:
            break

        read.id = str(idx)
        read.description = f'idx={idx}, strand=+, start={position}, end={position+length}'
        quality = {'phred_quality': [50] * len(read)}
        read.letter_annotations = quality  
        reads_list.append(read)

        step = int(get_normal_dist(step_mean, step_std))
        position += step
        idx += 1

    idx = len(reads_list)
    position = 0
    while position < stop:
        length = int(get_normal_dist(length_mean, length_std))
        if position + length < len(reference):
            read = reference_rc[position:position+length]
        else:
            break

        read.id = str(idx)
        read.description = f'idx={idx}, strand=-, start={len(reference)-position}, end={len(reference)-position-length}'
        quality = {'phred_quality': [50] * len(read)}
        read.letter_annotations = quality
        reads_list.append(read)

        step = int(get_normal_dist(step_mean, step_std))
        position += step
        idx += 1
    
    SeqIO.write(reads_list, out_path, 'fastq')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes reference path and path where to store the generated reads.')
    parser.add_argument('--ref', type=str, default='debug/reference.fasta', help='reference path')
    parser.add_argument('--out', type=str, default='debug/reads.fastq', help='path where to store the reads')
    parser.add_argument('--length-mean', type=int, default=20000, help='mean length of the simulated reads')
    parser.add_argument('--length-std', type=int, help='standard deviation in length of the simulated reads')
    parser.add_argument('--depth', type=int, default=20, help='sequencing depth to be simulated')
    args = parser.parse_args()
    main(args)

