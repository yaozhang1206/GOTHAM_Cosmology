"""
Prepare WST train/test data spanning the full 4000-sim extended-HOD set (grid=256 only).

Source data lives in two directories depending on isim:
  isim 0-1999:    /u/yzhang116/NN/inference/wide_hod_ext/WST_all_noise/
                  separate mock and truth files, WST computed (on DeltaAI) from
                  the positions of gal_pos_ext_40000_{mock,truth}.py, i.e. the
                  same LH_points_HOD_cosmo_ext_40000.txt HOD draws as Pk/Bk.
  isim 2000-3999: wide_hod_ext_40000/WST_all_noise/  (mock only, grid 256 only)

(The earlier isim 0-1999 files held mock+truth in one file and were painted
with LH_points_HOD_cosmo_ext_20000.txt, so their HOD did not match Pk/Bk.)

Train (isim NOT in saved_idx): mock WST only.
Test  (isim in saved_idx):     mock WST + truth WST.
saved_idx only contains values <= 1999, so all isim 2000-3999 are train.

Usage: python prepare_wst_train_test_40000.py [noise]
"""

import dill
import numpy as np
import os
import sys

noise = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0

ldir_stats_low  = '/u/yzhang116/NN/inference/wide_hod_ext/WST_all_noise/'
ldir_stats_high = '/work/hdd/bdne/yzhang116/quijote_galaxy/wide_hod_ext_40000/WST_all_noise/'
pos = 'rsd'

saved_idx = set(np.loadtxt("/u/yzhang116/NN/inference/quijote_saved_idx.txt", dtype=int).tolist())

nhod_LH_samp = 10
grid = 256
model = 'angular'
nsim = 4000

pk_train_mock = []
theta_train = []
nbar_train_mock = []

pk_test_mock = []
pk_test_truth = []
theta_test = []
nbar_test_mock = []
nbar_test_truth = []

for isim in range(nsim):
    ldir = ldir_stats_low if isim < 2000 else ldir_stats_high
    fname = ldir + f'WST_{model}_NGP_kc34pi_galnoise_{noise:.2f}_mock_galaxy_LH_{isim}.dill'

    try:
        saved_j = dill.load(open(fname, 'rb'))
        is_test = isim in saved_idx
        if is_test:
            # truth WST is in its own file (test sims are all isim < 2000)
            fname_t = ldir + f'WST_{model}_NGP_kc34pi_galnoise_{noise:.2f}_truth_galaxy_LH_{isim}.dill'
            saved_j = {**saved_j, **dill.load(open(fname_t, 'rb'))}

        for ihod in range(nhod_LH_samp):
            mock = np.concatenate([
                saved_j[f'rsd_s0_mock_{ihod}_{grid:d}'],
                saved_j[f'rsd_s1_mock_{ihod}_{grid:d}'],
                saved_j[f'rsd_s2_mock_{ihod}_{grid:d}'],
            ])
            theta_hod = list(saved_j[f'theta_hod_{ihod}'].values())
            theta_cosmo = list(saved_j[f'theta_cosmo_{ihod}'].values())
            theta_comb = np.array(theta_cosmo + theta_hod)
            nbar_mock = saved_j[f'galsum_mock_{ihod}']['number_density']

            if is_test:
                truth = np.concatenate([
                    saved_j[f'rsd_s0_truth_{ihod}_{grid:d}'],
                    saved_j[f'rsd_s1_truth_{ihod}_{grid:d}'],
                    saved_j[f'rsd_s2_truth_{ihod}_{grid:d}'],
                ])
                nbar_truth = saved_j[f'galsum_truth_{ihod}']['number_density']
                pk_test_mock.append(mock)
                pk_test_truth.append(truth)
                theta_test.append(theta_comb)
                nbar_test_mock.append(nbar_mock)
                nbar_test_truth.append(nbar_truth)
            else:
                pk_train_mock.append(mock)
                theta_train.append(theta_comb)
                nbar_train_mock.append(nbar_mock)

    except Exception as e:
        print(f"Error processing file for isim={isim}: {e}", flush=True)
        pass

pk_train_mock = np.array(pk_train_mock)
theta_train = np.array(theta_train)
nbar_train_mock = np.array(nbar_train_mock)

pk_test_mock = np.array(pk_test_mock)
pk_test_truth = np.array(pk_test_truth)
theta_test = np.array(theta_test)
nbar_test_mock = np.array(nbar_test_mock)
nbar_test_truth = np.array(nbar_test_truth)

output_train = {
    f'wst_mock_{grid:d}_all': pk_train_mock,
    'theta_all': theta_train,
    'nbar_mock': nbar_train_mock,
}

output_test = {
    f'wst_mock_{grid:d}_all': pk_test_mock,
    f'wst_truth_{grid:d}_all': pk_test_truth,
    'theta_all': theta_test,
    'nbar_mock': nbar_test_mock,
    'nbar_truth': nbar_test_truth,
}

print("data shapes:")
print("pk_train_mock:", pk_train_mock.shape)
print("theta_train:", theta_train.shape)
print("nbar_train_mock:", nbar_train_mock.shape)
print("pk_test_mock:", pk_test_mock.shape)
print("pk_test_truth:", pk_test_truth.shape)
print("theta_test:", theta_test.shape)
print("nbar_test_mock:", nbar_test_mock.shape)
print("nbar_test_truth:", nbar_test_truth.shape)

train_path = '/u/yzhang116/NN/inference/wide_hod_ext/WST_all_noise/' + f'WST_nbar_{model}_NGP_kc34pi_40000_galnoise_{noise:.2f}_train.dill'
test_path = '/u/yzhang116/NN/inference/wide_hod_ext/WST_all_noise/' + f'WST_nbar_{model}_NGP_kc34pi_40000_galnoise_{noise:.2f}_test.dill'

with open(train_path, 'wb') as f:
    dill.dump(output_train, f)

with open(test_path, 'wb') as f:
    dill.dump(output_test, f)

print(f"\nSaved train -> {train_path}")
print(f"Saved test  -> {test_path}")
