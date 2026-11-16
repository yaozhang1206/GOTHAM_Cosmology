"""
Prepare Bk/Qk train/test data from wide_hod_ext_40000 outputs.

Train  (simids 0-3999, NOT in saved_idx): mock Bk/Qk only.
Test   (simids in saved_idx):             mock Bk/Qk  +  truth Bk/Qk.

Usage: python prepare_bk_40000.py [noise]
"""

import dill
import numpy as np
import os
import sys

noise   = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
bk_dir  = '/work/hdd/bdne/yzhang116/quijote_galaxy/wide_hod_ext_40000/Bk_all_noise/'
pos     = 'rsd'
nhod    = 10
tags    = ['0p08', '0p16', '0p32']

saved_idx = set(np.loadtxt('/u/yzhang116/NN/inference/quijote_saved_idx.txt', dtype=int).tolist())


def concat_bk(saved, kind, ihod):
    """Concatenate Bk across the three k1 values for one HOD realisation."""
    return np.concatenate([saved[f'{pos}_Bk_{kind}_{tag}_{ihod}'] for tag in tags])


def concat_qk(saved, kind, ihod):
    return np.concatenate([saved[f'{pos}_Qk_{kind}_{tag}_{ihod}'] for tag in tags])


# --- train containers (mock only) ---
bk_train_mock   = []
qk_train_mock   = []
nbar_train_mock = []
theta_train     = []

# --- test containers (mock + truth) ---
bk_test_mock    = []
bk_test_truth   = []
qk_test_mock    = []
qk_test_truth   = []
nbar_test_mock  = []
nbar_test_truth = []
theta_test      = []

# -----------------------------------------------------------------------
# Training loop  --  all 4000 mock sims not in saved_idx
# -----------------------------------------------------------------------
for isim in range(4000):
    if isim in saved_idx:
        continue
    try:
        saved_j = dill.load(open(bk_dir + f'Bk_NGP_galnoise_{noise:.2f}_mock_LH_{isim}.dill', 'rb'))
        for ihod in range(nhod):
            bk_train_mock.append(concat_bk(saved_j, 'mock', ihod))
            qk_train_mock.append(concat_qk(saved_j, 'mock', ihod))
            nbar_train_mock.append(saved_j[f'galsum_mock_{ihod}']['number_density'])
            theta_hod   = list(saved_j[f'theta_hod_{ihod}'].values())
            theta_cosmo = list(saved_j[f'theta_cosmo_{ihod}'].values())
            theta_train.append(np.array(theta_cosmo + theta_hod))
    except Exception as e:
        print(f"train isim={isim}: {e}")

# -----------------------------------------------------------------------
# Test loop  --  saved_idx sims, load both mock and truth files
# -----------------------------------------------------------------------
for isim in sorted(saved_idx):
    try:
        saved_j_mock  = dill.load(open(bk_dir + f'Bk_NGP_galnoise_{noise:.2f}_mock_LH_{isim}.dill',  'rb'))
        saved_j_truth = dill.load(open(bk_dir + f'Bk_NGP_galnoise_{noise:.2f}_truth_LH_{isim}.dill', 'rb'))
        for ihod in range(nhod):
            bk_test_mock.append(concat_bk(saved_j_mock,  'mock',  ihod))
            bk_test_truth.append(concat_bk(saved_j_truth, 'truth', ihod))
            qk_test_mock.append(concat_qk(saved_j_mock,  'mock',  ihod))
            qk_test_truth.append(concat_qk(saved_j_truth, 'truth', ihod))
            nbar_test_mock.append(saved_j_mock[f'galsum_mock_{ihod}']['number_density'])
            nbar_test_truth.append(saved_j_truth[f'galsum_truth_{ihod}']['number_density'])
            theta_hod   = list(saved_j_mock[f'theta_hod_{ihod}'].values())
            theta_cosmo = list(saved_j_mock[f'theta_cosmo_{ihod}'].values())
            theta_test.append(np.array(theta_cosmo + theta_hod))
    except Exception as e:
        print(f"test  isim={isim}: {e}")

# -----------------------------------------------------------------------
# Stack and report
# -----------------------------------------------------------------------
bk_train_mock   = np.array(bk_train_mock)
qk_train_mock   = np.array(qk_train_mock)
nbar_train_mock = np.array(nbar_train_mock)
theta_train     = np.array(theta_train)

bk_test_mock    = np.array(bk_test_mock)
bk_test_truth   = np.array(bk_test_truth)
qk_test_mock    = np.array(qk_test_mock)
qk_test_truth   = np.array(qk_test_truth)
nbar_test_mock  = np.array(nbar_test_mock)
nbar_test_truth = np.array(nbar_test_truth)
theta_test      = np.array(theta_test)

print("=== data shapes ===")
print("bk_train_mock:  ", bk_train_mock.shape)
print("qk_train_mock:  ", qk_train_mock.shape)
print("theta_train:    ", theta_train.shape)
print("nbar_train_mock:", nbar_train_mock.shape)
print("bk_test_mock:   ", bk_test_mock.shape)
print("bk_test_truth:  ", bk_test_truth.shape)
print("qk_test_mock:   ", qk_test_mock.shape)
print("qk_test_truth:  ", qk_test_truth.shape)
print("theta_test:     ", theta_test.shape)
print("nbar_test_mock: ", nbar_test_mock.shape)
print("nbar_test_truth:", nbar_test_truth.shape)

output_train = {
    'bk_all_mock':  bk_train_mock,
    'qk_all_mock':  qk_train_mock,
    'theta_all':    theta_train,
    'nbar_mock':    nbar_train_mock,
}

output_test = {
    'bk_all_mock':  bk_test_mock,
    'bk_all_truth': bk_test_truth,
    'qk_all_mock':  qk_test_mock,
    'qk_all_truth': qk_test_truth,
    'theta_all':    theta_test,
    'nbar_mock':    nbar_test_mock,
    'nbar_truth':   nbar_test_truth,
}

train_path = bk_dir + f'Bk_NGP_galnoise_{noise:.2f}_200c_train.dill'
test_path  = bk_dir + f'Bk_NGP_galnoise_{noise:.2f}_200c_test.dill'

with open(train_path, 'wb') as f:
    dill.dump(output_train, f)
with open(test_path, 'wb') as f:
    dill.dump(output_test, f)

print(f"\nSaved train → {train_path}")
print(f"Saved test  → {test_path}")
