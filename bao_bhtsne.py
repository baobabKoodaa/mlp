#!/usr/bin/env python
# https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf
# Usage:
# 
'''
The output will not be normalised, maybe the below one-liner is of interest?:

    python -c 'import numpy;  from sys import stdin, stdout;
        d = numpy.loadtxt(stdin); d -= d.min(axis=0); d /= d.max(axis=0);
        numpy.savetxt(stdout, d, fmt="%.8f", delimiter="\t")'

Authors:     Pontus Stenetorp    <pontus stenetorp se>
             Philippe Remy       <github: philipperemy>
			 Modified by bear330 from the Numerai Slack channel
			 Modified by Marianne from the Numerai Slack channel
Version:    2016-03-08
'''

# Copyright (c) 2013, Pontus Stenetorp <pontus stenetorp se>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

from argparse import ArgumentParser, FileType
from os.path import abspath, dirname, isfile, join as path_join
from shutil import rmtree
from struct import calcsize, pack, unpack
from subprocess import Popen
from sys import stderr, stdin, stdout
from tempfile import mkdtemp
from platform import system
from os import devnull
import numpy as np
import os, sys

### Constants
IS_WINDOWS = True if system() == 'Windows' else False
BH_TSNE_BIN_PATH = path_join(dirname(__file__), 'bh_tsne.exe') if IS_WINDOWS else path_join(dirname(__file__), 'bh_tsne')
assert isfile(BH_TSNE_BIN_PATH), ('Unable to find the bh_tsne binary in the '
    'same directory as this script, have you forgotten to compile it?: {}'
    ).format(BH_TSNE_BIN_PATH)
# Default hyper-parameter values from caioaao in the Numerai Slack channel
DEFAULT_NO_DIMS = 3
INITIAL_DIMENSIONS = 50
DEFAULT_PERPLEXITY = 50
DEFAULT_THETA = 0.5 # 0.0 for theta is equivalent to vanilla t-SNE
EMPTY_SEED = 984
DEFAULT_USE_PCA = True
DEFAULT_MAX_ITERATIONS = 1000

###

def init_bh_tsne(input_arr, workdir, no_dims=DEFAULT_NO_DIMS, initial_dims=INITIAL_DIMENSIONS, perplexity=DEFAULT_PERPLEXITY,
            theta=DEFAULT_THETA, randseed=EMPTY_SEED, verbose=False, use_pca=DEFAULT_USE_PCA, max_iter=DEFAULT_MAX_ITERATIONS):

    samples = input_arr # np.asarray(samples, dtype='float64')

    if use_pca:
        samples = samples - np.mean(samples, axis=0)
        cov_x = np.dot(np.transpose(samples), samples)
        [eig_val, eig_vec] = np.linalg.eig(cov_x)

        # sorting the eigen-values in the descending order
        eig_vec = eig_vec[:, eig_val.argsort()[::-1]]

        if initial_dims > len(eig_vec):
            initial_dims = len(eig_vec)

        # truncating the eigen-vectors matrix to keep the most important vectors
        eig_vec = eig_vec[:, :initial_dims]
        samples = np.dot(samples, eig_vec)

    # Assume that the dimensionality of the first sample is representative for the whole batch
    sample_dim = len(samples[0])
    sample_count = len(samples)

    # Note: The binary format used by bh_tsne is roughly the same as for vanilla tsne
    with open(path_join(workdir, 'data.dat'), 'wb') as data_file:
        # Write the bh_tsne header
        data_file.write(pack('iiddii', sample_count, sample_dim, theta, perplexity, no_dims, max_iter))
        # Then write the data
        for sample in samples:
            data_file.write(pack('{}d'.format(len(sample)), *sample))
        # Write random seed
        data_file.write(pack('i', randseed))

def _read_unpack(fmt, fh):
    return unpack(fmt, fh.read(calcsize(fmt)))

def bh_tsne(workdir, verbose=False):

    # Call bh_tsne and let it do its thing
    with open(devnull, 'w') as dev_null:
        bh_tsne_p = Popen((abspath(BH_TSNE_BIN_PATH), ), cwd=workdir,
                # bh_tsne is very noisy on stdout, tell it to use stderr
                #   if it is to print any output
                stdout=stderr if verbose else dev_null)
        bh_tsne_p.wait()
        assert not bh_tsne_p.returncode, ('ERROR: Call to bh_tsne exited '
                'with a non-zero return code exit status, please ' +
                ('enable verbose mode and ' if not verbose else '') +
                'refer to the bh_tsne output for further details')

    # Read and pass on the results
    with open(path_join(workdir, 'result.dat'), 'rb') as output_file:
        # The first two integers are just the number of samples and the
        #   dimensionality
        result_samples, result_dims = _read_unpack('ii', output_file)
        # Collect the results, but they may be out of order
        results = [_read_unpack('{}d'.format(result_dims), output_file)
            for _ in range(result_samples)]
        # Now collect the landmark data so that we can return the data in
        #   the order it arrived
        results = [(_read_unpack('i', output_file), e) for e in results]
        # Put the results in order and yield it
        results.sort()
        for _, result in results:
            yield result
        # The last piece of data is the cost for each sample, we ignore it
        #read_unpack('{}d'.format(sample_count), output_file)

def run_bh_tsne(input_arr, no_dims=DEFAULT_NO_DIMS, perplexity=DEFAULT_PERPLEXITY, theta=DEFAULT_THETA, randseed=EMPTY_SEED, verbose=False, initial_dims=INITIAL_DIMENSIONS, use_pca=DEFAULT_USE_PCA, max_iter=DEFAULT_MAX_ITERATIONS):
    '''
    Run TSNE based on the Barnes-HT algorithm

    Parameters:
    ----------
    data: numpy.array
        The data used to run TSNE, one sample per row
    no_dims: int
    perplexity: int
    randseed: int
    theta: float
    initial_dims: int
    verbose: boolean
    use_pca: boolean
    max_iter: int
    '''

    # bh_tsne works with fixed input and output paths, give it a temporary
    #   directory to work in so we don't clutter the filesystem
    tmp_dir_path = mkdtemp()

    # Load data in another process to free memory for actual bh_tsne calculation
    #from multiprocessing import Process
    #p = Process(target=init_bh_tsne, args=(input_arr, tmp_dir_path), kwargs=dict(no_dims=no_dims, perplexity=perplexity, theta=theta, randseed=randseed,verbose=verbose, initial_dims=initial_dims, use_pca=use_pca, max_iter=max_iter))
    #p.start()
    #p.join()
    init_bh_tsne(input_arr, tmp_dir_path, no_dims=no_dims, perplexity=perplexity, theta=theta, randseed=randseed,verbose=verbose, initial_dims=initial_dims, use_pca=use_pca, max_iter=max_iter)
    res = []
    
    raw_results = bh_tsne(tmp_dir_path, verbose)
    for result in raw_results:
        sample_res = []
        for r in result:
            sample_res.append(r)
        res.append(sample_res)
    rmtree(tmp_dir_path)
    return np.asarray(res, dtype='float64')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   

def process_results(in_arr, out_filename):
    # Write results into a CSV file
    out_file = open(out_filename, 'w')
    for result in in_arr:
        fmt = ''
        for i in range(1, len(result)):
            fmt = fmt + '{},'
        fmt = fmt + '{}\n'
        out_file.write(fmt.format(*result))
    out_file.close()

    # Plot results
    N = M = 0
    all_data = {}
    return
    for i, line in enumerate(open(out_filename)):
        if i >= len(label):
            break
        vec = line.strip().split(',')
        all_data.setdefault(label[i - 1], []).append(
            (float(vec[-3]), float(vec[-2]), float(vec[-1]))) # TODO: FIX TO WORK WITH 2D AS WELL
        N += 1
        M = len(vec)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(all_data)))

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')    
    
    for color, ll in zip(colors, sorted(all_data.keys())):
        x = [t[0] for t in all_data[ll]]
        y = [t[1] for t in all_data[ll]]
        z = [t[2] for t in all_data[ll]]
        plt.plot(x, y, z, '.', color = color, markersize = 1)
    plt.savefig('%s.png' % out_filename, dpi=500)
