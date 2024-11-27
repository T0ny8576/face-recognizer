import os
import numpy as np
import matplotlib.pyplot as plt

BASEDIR = os.path.dirname(os.path.abspath(__file__))


def plot_comparison(data_to_compare, config_names, group_xticks, title, output_file):
    labels = ["detect", "align", "represent", "classify"]
    colors = ["orange", "blue", "green", "red"]
    y_data = np.stack([data for data in data_to_compare]).T
    fig, ax = plt.subplots()
    ax.set_ylabel('Time (ms)')
    ax.set_title(title)

    # Group by 2
    group_size = len(group_xticks)
    width = 0.2 - 0.01
    offset = 0.2
    num_groups = len(config_names) // group_size
    x = np.repeat(np.arange(num_groups), group_size)
    offsets = np.array([-1.5*offset, -0.5*offset, 0.5*offset, 1.5*offset] * num_groups)
    x_grouped = x + offsets
    plt.xticks(x_grouped, labels=config_names, rotation=35, ha='right')
    bottom = np.zeros(len(config_names))
    for i in range(len(labels)):
        p = ax.bar(x_grouped, y_data[i], label=labels[i], capsize=3, width=width,
                   bottom=bottom, color=colors[i])
        bottom += y_data[i]
    for x_pos in range(len(x_grouped)):
        ax.text(x_grouped[x_pos], bottom[x_pos] + 1., round(bottom[x_pos]), ha='center', fontsize=10)

    # Reorder the labels in the legend
    handles, lbs = ax.get_legend_handles_labels()
    handles.reverse()
    lbs.reverse()
    ax.legend(handles, lbs, loc="upper left")

    sec = ax.secondary_xaxis(location=0)
    sec.set_xticks(np.arange(len(group_xticks)) - offset,
                   labels=["\n\n\n\n\n" + xtick for xtick in group_xticks])
    sec.tick_params('x', length=0)

    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    data_240p = [0.011, 0.003, 0.012, 0.001]
    data_480p = [0.013, 0.003, 0.009, 0.001]
    data_720p = [0.020, 0.003, 0.010, 0.001]
    data_1080p = [0.038, 0.003, 0.010, 0.001]

    data_240p_up = [0.009, 0.003, 0.008, 0.001]
    data_480p_up = [0.031, 0.003, 0.010, 0.001]
    data_720p_up = [0.092, 0.003, 0.010, 0.001]
    data_1080p_up = [0.204, 0.004, 0.009, 0.001]

    data_240p_hog = [0.022, 0.002, 0.010, 0.001]
    data_480p_hog = [0.050, 0.002, 0.009, 0.001]
    data_720p_hog = [0.100, 0.004, 0.010, 0.001]
    data_1080p_hog = [0.222, 0.003, 0.010, 0.001]

    data_240p_hog_up = [0.037, 0.002, 0.009, 0.001]
    data_480p_hog_up = [0.140, 0.003, 0.009, 0.001]
    data_720p_hog_up = [0.424, 0.003, 0.010, 0.001]
    data_1080p_hog_up = [0.940, 0.020, 0.011, 0.001]

    data_to_plot = np.array([data_240p_hog_up, data_240p_hog, data_240p_up, data_240p,
                             data_480p_hog_up, data_480p_hog, data_480p_up, data_480p,
                             data_720p_hog_up, data_720p_hog, data_720p_up, data_720p,
                             data_1080p_hog_up, data_1080p_hog, data_1080p_up, data_1080p]) * 1000
    trial_names = ["HOG upsample", "HOG", "CNN upsample", "CNN",
                   "HOG upsample", "HOG", "CNN upsample", "CNN",
                   "HOG upsample", "HOG", "CNN upsample", "CNN",
                   "HOG upsample", "HOG", "CNN upsample", "CNN"]
    group_names = ["240p", "480p", "720p", "1080p"]
    plot_comparison(data_to_plot, trial_names, group_names, "OpenFace Server Processing Time Breakdown",
                    os.path.join(BASEDIR, "openface_server_breakdown.png"))
