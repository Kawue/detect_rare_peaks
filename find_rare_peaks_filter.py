import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 200

def normalize(data):
    mmin = np.amin(data)
    mmax = np.amax(data)
    norm = (data - mmin) / (mmax - mmin)
    return norm

parser = argparse.ArgumentParser(description="...")
parser.add_argument("-r", "--readfiles", dest='readfiles', type=str,  nargs="+", required=True, help="Path to hdf5 files.")
parser.add_argument("-s", "--savepath", dest='savepath', type=str, required=True, help="Savepath")
parser.add_argument("-t", "--threshold", dest='threshold', type=float, required=True, help="Filter treshold between 0 and 1.")
parser.add_argument("-b", "--binning", dest='binning', type=int, required=True, help="Number of bins for peak analysis.")
parser.add_argument("-f", "--filter_t", dest='filter', type=int, required=True, help="Height of bins for selection.")
args = parser.parse_args()

readfiles = args.readfiles
filter_t = args.threshold
binning = args.binning
binning_t = args.filter
savepath = args.savepath
savepath_plots = os.path.join(savepath, "plots")
if not os.path.exists(savepath):
    os.makedirs(savepath)
if not os.path.exists(savepath_plots):
    os.makedirs(savepath_plots)
files = []
filenames = []


for i, f in enumerate(readfiles):
    files.append(pd.read_hdf(f))
    filenames.append(os.path.basename(f))
    print(f"Dimensions: {files[i].shape}")
print()
print("------------------")
print()

# mean and max specs
mean_specs = []
max_specs = []
for f in files:
    mean_spec = normalize(f.mean(axis=0))
    max_spec = normalize(f.max(axis=0))
    mean_specs.append(mean_spec)
    max_specs.append(max_spec)


# checkup plot
for i,_ in enumerate(files):
    plt.figure()
    plt.title(f"Mean Spec File Nr.{i} ({filenames[i]})")
    plt.stem(files[i].columns, mean_specs[i], markerfmt='', basefmt='', bottom=-0.01 , use_line_collection=True)
    plt.savefig(os.path.join(savepath_plots, f"mean_spec_file_nr{i}.png"))
    plt.close()

    plt.figure()
    plt.title(f"Max Spec File Nr.{i} ({filenames[i]})")
    plt.stem(files[i].columns, max_specs[i], markerfmt='', basefmt='', bottom=-0.01 , use_line_collection=True)
    plt.savefig(os.path.join(savepath_plots, f"max_spec_file_nr{i}.png"))
    plt.close()


# filter picked data
mean_specs_filtered_idxs = []
max_specs_filtered_idxs = []
mean_specs_filtered_mzs = []
max_specs_filtered_mzs = []
for i,_ in enumerate(mean_specs):
    mean_filter_idx = np.where(mean_specs[i] >= filter_t)[0]
    max_filter_idx = np.where(max_specs[i] >= filter_t)[0]
    
    mean_specs_filtered_idxs.append(mean_filter_idx)
    max_specs_filtered_idxs.append(max_filter_idx)
    
    mzs = files[i].columns.astype(float)
    mean_specs_filtered_mzs.append(mzs[mean_filter_idx])
    max_specs_filtered_mzs.append(mzs[max_filter_idx])

for i, _ in enumerate(files):
    print(f"File Nr.{i} ({filenames[i]}) -- Mean")
    print(f"Number of Peaks before Thresholding: {len(mean_specs[i])}")
    print(f"Number of Peaks after Thresholding:  {len(mean_specs_filtered_idxs[i])}")
    print()
print()

for i, _ in enumerate(files):
    print(f"File Nr.{i} ({filenames[i]}) -- Max")
    print(f"Number of Peaks before Thresholding: {len(max_specs[i])}")
    print(f"Number of Peaks after Thresholding:  {len(max_specs_filtered_idxs[i])}")
    print()
print("------------------")
print()


# histogram filter
mzs = [f.columns.astype(float) for f in files]
min_mz = np.amin(mzs)
max_mz = np.amax(mzs)
mean_mz_hist, mean_mz_bins = np.histogram([x for sublist in mean_specs_filtered_mzs for x in sublist], bins=binning, range=(min_mz, max_mz))
max_mz_hist, max_mz_bins =np.histogram([x for sublist in max_specs_filtered_mzs for x in sublist], bins=binning, range=(min_mz, max_mz))

# plot mean hist
mean_width = 0.8 * (mean_mz_bins[1] - mean_mz_bins[0])
mean_centers = (mean_mz_bins[:-1] + mean_mz_bins[1:]) / 2
plt.figure()
plt.title(f"Histogram Mean")
plt.bar(mean_centers, mean_mz_hist, align="center", width=mean_width)
plt.savefig(os.path.join(savepath_plots, "mean_histogram.png"))
plt.close()

# plot max hist
max_width = 0.8 * (max_mz_bins[1] - max_mz_bins[0])
max_centers = (max_mz_bins[:-1] + max_mz_bins[1:]) / 2
plt.figure()
plt.title(f"Histogram Max")
plt.bar(max_centers, max_mz_hist, align="center", width=max_width)
plt.savefig(os.path.join(savepath_plots, "max_histogram.png"))
plt.close()


# filter mean
index_mz_bins_mean = np.where((0<mean_mz_hist)*(mean_mz_hist<=binning_t))[0]
bin_edge_tuples_mean = list(zip(mean_mz_bins[:-1], mean_mz_bins[1:]))
rare_tupels_mean = [(bin_edge_tuples_mean[i][0], bin_edge_tuples_mean[i][1]) for i in index_mz_bins_mean]

rare_mzs_mean = []
for mz_list in mean_specs_filtered_mzs:
    rare_mz = []
    for i in mz_list:
        for x,y in rare_tupels_mean:
            if x <= i <= y:
                rare_mz.append(i)
    rare_mzs_mean.append(rare_mz)

for i, _ in enumerate(files):
    print(f"File Nr.{i} ({filenames[i]}) -- Mean")
    print(f"Number of Peaks before Thresholding: {len(mean_specs[i])}")
    print(f"Number of Peaks after Thresholding:  {len(mean_specs_filtered_mzs[i])}")
    print(f"Number of Peaks after Filtering:     {len(rare_mzs_mean[i])}")
    #print(rare_mzs_mean[i])
    print()
print()


# filter max
index_mz_bins_max = np.where((0<max_mz_hist)*(max_mz_hist<=binning_t))[0]
bin_edge_tuples_max = list(zip(max_mz_bins[:-1], max_mz_bins[1:]))
rare_tupels_max = [(bin_edge_tuples_max[i][0], bin_edge_tuples_max[i][1]) for i in index_mz_bins_max]

rare_mzs_max = []
for mz_list in max_specs_filtered_mzs:
    rare_mz = []
    for i in mz_list:
        for x,y in rare_tupels_max:
            if x <= i <= y:
                rare_mz.append(i)
    rare_mzs_max.append(rare_mz)

for i, _ in enumerate(files):
    print(f"File Nr.{i} ({filenames[i]}) -- Max")
    print(f"Number of Peaks before Thresholding: {len(max_specs[i])}")
    print(f"Number of Peaks after Thresholding:  {len(max_specs_filtered_mzs[i])}")
    print(f"Number of Peaks after Filtering:     {len(rare_mzs_max[i])}")
    #print(rare_mzs_max[i])
    print()



# create datasets mean
for i, _ in enumerate(files):
    if len(rare_mzs_mean[i]) > 0:
        name = os.path.basename(readfiles[i]).split(".h5")[0] + "rare_peaks_mean"
        #rare_dframe = files[i][files[i].columns[rare_mzs_mean[i]]]
        rare_dframe = files[i][rare_mzs_mean[i]]
        print(rare_dframe.shape)
        rare_dframe.to_hdf(os.path.join(savepath, name+".h5"), key=name, complib="blosc", complevel=9)

# create datasets max
for i, _ in enumerate(files):
    if len(rare_mzs_max[i]) > 0:
        name = os.path.basename(readfiles[i]).split(".h5")[0] + "rare_peaks_max"
        #rare_dframe = files[i][files[i].columns[rare_mzs_max[i]]]
        rare_dframe = files[i][rare_mzs_max[i]]
        print(rare_dframe.shape)
        rare_dframe.to_hdf(os.path.join(savepath, name+".h5"), key=name, complib="blosc", complevel=9)
