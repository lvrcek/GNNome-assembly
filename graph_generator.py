import os
import random
import subprocess


def generate_pacbio(num_graphs, ref_path, out_dir):
    pbsim_dir = os.path.abspath('/home/lovro/Software/PBSIM')
    pbsim_path = os.path.join(pbsim_dir, 'src/pbsim')
    model_path = os.path.join(pbsim_dir, 'data/model_qc_clr')
    outdir_path = os.path.abspath(out_dir)
    ref_path = os.path.abspath(ref_path)
    depth_list = [20, 25, 30]
    length_mean = 10000
    acc_mean_list = [0.96, 0.97, 0.98]
    acc_min_list = [0.92, 0.93]
    if not os.path.isdir(outdir_path):
        os.mkdir(outdir_path)

    with open('read_info.txt', 'w') as f:
        for n in range(num_graphs):
            depth = random.choice(depth_list)
            acc_mean = random.choice(acc_mean_list)
            acc_min = random.choice(acc_min_list)

            command = f'{pbsim_path} --data-type CLR --model_qc {model_path} '
            command += f'--prefix {n} --depth {depth} --length-mean {length_mean} '
            command += f'--accuracy-mean {acc_mean} --accuracy-min {acc_min} '
            command += f'{ref_path}'

            subprocess.run(command, shell=True, cwd=outdir_path)
            f.write(f'{n}: depth = {depth}, acc_mean={acc_mean}, acc_min={acc_min}\n')

        subprocess.run(f'rm *.ref', shell=True, cwd=outdir_path)
        subprocess.run(f'rm *.maf', shell=True, cwd=outdir_path)


def generate_ont(num_graphs, ref_path):
    pass


if __name__ == '__main__':
    reference = 'data/references/lambda_reference.fasta'
    outdir = 'data/debug/pbsim'
    generate_pacbio(5, reference, outdir)
