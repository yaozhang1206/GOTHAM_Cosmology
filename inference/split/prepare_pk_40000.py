"""
Prepare Pk train/test data from wide_hod_ext_40000 outputs.

Train  (simids 0-3999, NOT in saved_idx): mock Pk only.
Test   (simids in saved_idx):             mock Pk  +  truth Pk.

Usage: python prepare_pk_40000.py [noise]
"""

import dill
import numpy as np
import os
import sys

noise    = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
pk_dir   = '/work/hdd/bdne/yzhang116/quijote_galaxy/wide_hod_ext_40000/Pk_all_noise/'
pos      = 'rsd'
nhod     = 10

saved_idx = set(np.loadtxt('/u/yzhang116/NN/inference/quijote_saved_idx.txt', dtype=int).tolist())

# --- train containers (mock only) ---
pk_train_mock   = []
nbar_train_mock = []
theta_train     = []

# --- test containers (mock + truth) ---
pk_test_mock    = []
pk_test_truth   = []
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
        saved_k = dill.load(open(pk_dir + f'Pk_NGP_galnoise_{noise:.2f}_mock_LH_{isim}.dill', 'rb'))
        for ihod in range(nhod):
            pk_train_mock.append(saved_k[f'{pos}_Pk_mock_{ihod}'])
            nbar_train_mock.append(saved_k[f'galsum_mock_{ihod}']['number_density'])
            theta_hod   = list(saved_k[f'theta_hod_{ihod}'].values())
            theta_cosmo = list(saved_k[f'theta_cosmo_{ihod}'].values())
            theta_train.append(np.array(theta_cosmo + theta_hod))
    except Exception as e:
        print(f"train isim={isim}: {e}")

# -----------------------------------------------------------------------
# Test loop  --  saved_idx sims, load both mock and truth files
# -----------------------------------------------------------------------
for isim in sorted(saved_idx):
    try:
        saved_k_mock  = dill.load(open(pk_dir + f'Pk_NGP_galnoise_{noise:.2f}_mock_LH_{isim}.dill',  'rb'))
        saved_k_truth = dill.load(open(pk_dir + f'Pk_NGP_galnoise_{noise:.2f}_truth_LH_{isim}.dill', 'rb'))
        for ihod in range(nhod):
            pk_test_mock.append(saved_k_mock[f'{pos}_Pk_mock_{ihod}'])
            pk_test_truth.append(saved_k_truth[f'{pos}_Pk_truth_{ihod}'])
            nbar_test_mock.append(saved_k_mock[f'galsum_mock_{ihod}']['number_density'])
            nbar_test_truth.append(saved_k_truth[f'galsum_truth_{ihod}']['number_density'])
            theta_hod   = list(saved_k_mock[f'theta_hod_{ihod}'].values())
            theta_cosmo = list(saved_k_mock[f'theta_cosmo_{ihod}'].values())
            theta_test.append(np.array(theta_cosmo + theta_hod))
    except Exception as e:
        print(f"test  isim={isim}: {e}")

# -----------------------------------------------------------------------
# Stack and report
# -----------------------------------------------------------------------
pk_train_mock   = np.array(pk_train_mock)
nbar_train_mock = np.array(nbar_train_mock)
theta_train     = np.array(theta_train)

pk_test_mock    = np.array(pk_test_mock)
pk_test_truth   = np.array(pk_test_truth)
nbar_test_mock  = np.array(nbar_test_mock)
nbar_test_truth = np.array(nbar_test_truth)
theta_test      = np.array(theta_test)

print("=== data shapes ===")
print("pk_train_mock:  ", pk_train_mock.shape)
print("theta_train:    ", theta_train.shape)
print("nbar_train_mock:", nbar_train_mock.shape)
print("pk_test_mock:   ", pk_test_mock.shape)
print("pk_test_truth:  ", pk_test_truth.shape)
print("theta_test:     ", theta_test.shape)
print("nbar_test_mock: ", nbar_test_mock.shape)
print("nbar_test_truth:", nbar_test_truth.shape)

# k-axis from the last successfully loaded mock file
k = saved_k_mock['k_Pk_0']

output_train = {
    'pk_all_mock':  pk_train_mock,
    'k':            k,
    'theta_all':    theta_train,
    'nbar_mock':    nbar_train_mock,
}

output_test = {
    'pk_all_mock':  pk_test_mock,
    'pk_all_truth': pk_test_truth,
    'k':            k,
    'theta_all':    theta_test,
    'nbar_mock':    nbar_test_mock,
    'nbar_truth':   nbar_test_truth,
}

train_path = pk_dir + f'Pk_nbar_NGP_galnoise_{noise:.2f}_200c_train.dill'
test_path  = pk_dir + f'Pk_nbar_NGP_galnoise_{noise:.2f}_200c_test.dill'

with open(train_path, 'wb') as f:
    dill.dump(output_train, f)
with open(test_path, 'wb') as f:
    dill.dump(output_test, f)

print(f"\nSaved train → {train_path}")
print(f"Saved test  → {test_path}")
