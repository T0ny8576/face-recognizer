import os
import numpy as np
import matplotlib.pyplot as plt

NUM_FRAME_DISCARDED = 4
NUM_TOTAL_FRAME_DEFAULT = 500 + NUM_FRAME_DISCARDED


def plot_histogram(data, xlabel, ylabel, fname, bins=40, dpi=300, x_range=None):
    fig, ax = plt.subplots()
    ax.hist(data, bins=bins, range=x_range)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    agg_text = "mean = {:.2f}\nmedian = {:.2f}\nstd = {:.2f}".format(np.mean(data), np.median(data), np.std(data))
    ax.text(0.75, 0.85, agg_text, transform=ax.transAxes, bbox=dict(fc='white', ec='black', alpha=0.5))
    fig.savefig(fname, dpi=dpi)
    plt.close(fig)


def plot_individual_timeline(per_frame_data, ylabel, fname, frame_to_plot=None):
    fig, ax = plt.subplots()
    ax.set_xlabel('Frame Id')
    ax.set_ylabel(ylabel)
    if frame_to_plot is None:
        frame_to_plot = len(per_frame_data)
    ax.set_xlim(0, frame_to_plot)
    fig.set_figwidth(60)
    p = ax.plot(np.arange(frame_to_plot), per_frame_data[:frame_to_plot], '.-')
    for i in range(frame_to_plot):
        ax.text(i, per_frame_data[:frame_to_plot][i], int(per_frame_data[:frame_to_plot][i]), ha='center', fontsize=6)
    agg_text = "mean = {:.2f}\nmedian = {:.2f}\nstd = {:.2f}".format(
        np.mean(per_frame_data), np.median(per_frame_data), np.std(per_frame_data))
    ax.text(0.9, 0.85, agg_text, transform=ax.transAxes, bbox=dict(fc='white', ec='black', alpha=0.5))
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_engine(engine_log_file, log_dir, plot_detailed_figures):
    (engine_start, engine_done, engine_pre, engine_proc,
     engine_post, engine_total, engine_idle) = (
        tuple(np.squeeze(arr) for arr in np.split(np.zeros((7, NUM_TOTAL_FRAME_DEFAULT)), 7, axis=0)))

    with open(os.path.join(log_dir, engine_log_file), "r") as server_log:
        server_lines = server_log.readlines()
        for sl in server_lines:
            if len(sl.strip()) > 0:
                sls = sl.split(", ")
                if sls[0] in ["#" + str(i + 1) for i in range(NUM_FRAME_DISCARDED)]:
                    continue
                idx = int(sls[0].replace("#", "")) - NUM_FRAME_DISCARDED - 1
                engine_start[idx] = float(sls[1].replace("time = ", "")) * 1000
                engine_done[idx] = float(sls[2].replace("done = ", "")) * 1000
                engine_pre[idx] = float(sls[3].replace("pre = ", "").replace(" ms", ""))
                engine_proc[idx] = float(sls[4].replace("proc = ", "").replace(" ms", ""))
                engine_post[idx] = float(sls[7].replace("post = ", "").replace(" ms", ""))
                engine_idle[idx] = float(sls[8].replace("idle = ", "").replace(" ms", ""))
                engine_total[idx] = engine_done[idx] - engine_start[idx]

    num_frames = sum(engine_done > 0)
    print(num_frames)
    engine_start = engine_start[:num_frames]
    engine_done = engine_done[:num_frames]
    engine_pre = engine_pre[:num_frames]
    engine_proc = engine_proc[:num_frames]
    engine_post = engine_post[:num_frames]
    engine_total = engine_total[:num_frames]
    engine_idle = engine_idle[1:num_frames]

    stacked = np.stack([engine_pre, engine_proc, engine_post])

    if plot_detailed_figures:
        """ Individual histograms
        """
        plot_histogram(engine_pre, 'Time (ms)', 'Frame Count', os.path.join(log_dir, "engine_pre.png"))
        plot_histogram(engine_proc, 'Time (ms)', 'Frame Count', os.path.join(log_dir, "engine_proc.png"))
        plot_histogram(engine_post, 'Time (ms)', 'Frame Count', os.path.join(log_dir, "engine_post.png"))
        plot_histogram(engine_idle, 'Time (ms)', 'Frame Count', os.path.join(log_dir, "engine_idle.png"))
        plot_histogram(engine_total, 'Time (ms)', 'Frame Count', os.path.join(log_dir, "engine_total.png"))

        """ Per-frame individual timeline 
        """
        plot_individual_timeline(engine_proc, 'Server Processing Time (ms)',
                                 os.path.join(log_dir, "engine_proc_timeline.png"))

        """ Per-frame stacked timeline
        """
        labels = ["engine_pre", "engine_proc", "engine_post"]
        fig, ax = plt.subplots()
        ax.set_xlabel('Frame Id')
        num_chosen = num_frames
        ax.set_xlim(0, num_chosen)
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Server Engine Processing Time on Each Frame')
        fig.set_figwidth(50)

        bottom = np.zeros(num_chosen)
        for i in range(len(labels)):
            p = ax.bar(np.arange(num_chosen), stacked[i][:num_chosen], width=1., label=labels[i], bottom=bottom)
            bottom += stacked[i][:num_chosen]

        # Reorder the labels in the legend
        handles, lbs = ax.get_legend_handles_labels()
        handles.reverse()
        lbs.reverse()
        ax.legend(handles, lbs, loc="upper right")

        agg_text = "mean = {:.2f}\nmedian = {:.2f}\nstd = {:.2f}".format(
            np.mean(engine_total), np.median(engine_total), np.std(engine_total))
        ax.text(0.9, 0.85, agg_text, transform=ax.transAxes, bbox=dict(fc='white', ec='black', alpha=0.5))
        fig.savefig(os.path.join(log_dir, "engine_total_timeline_stacked.png"), dpi=600, bbox_inches="tight")
        plt.close(fig)

    return (np.mean(stacked, axis=1), np.median(stacked, axis=1), np.std(stacked, axis=1),
            np.percentile(stacked, 2.5, axis=1), np.percentile(stacked, 97.5, axis=1))


if __name__ == "__main__":
    BASEDIR = os.path.dirname(os.path.abspath(__file__))
    LOGDIR = os.path.join(BASEDIR, "logs/", "test_fixed_1080p_new/")
    ENGINE_LOG = "Engine-Log.txt"
    plot_engine(ENGINE_LOG, LOGDIR, True)
