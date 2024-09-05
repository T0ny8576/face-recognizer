import os
import numpy as np
import matplotlib.pyplot as plt

NUM_TOTAL_FRAME_DEFAULT = 501
NUM_FRAME_DISCARDED = 4


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


def plot_trial(server_log_file, client_log_file, log_dir, plot_detailed_figures):
    (client_gen, client_send, client_recv, client_done,
     server_recv, server_done, server_pre, server_detect,
     server_repr, server_infer, server_post, server_total, server_wait) = (
        tuple(np.squeeze(arr) for arr in np.split(np.zeros((13, NUM_TOTAL_FRAME_DEFAULT)), 13, axis=0)))
    with open(os.path.join(log_dir, client_log_file), "r") as client_log:
        client_lines = client_log.readlines()
        for cl in client_lines:
            cls = cl.split()
            if cls[0] in [str(i + 1) for i in range(NUM_FRAME_DISCARDED)]:
                continue
            if "Gen" in cl and "Failed" not in cl:
                client_gen[int(cls[0]) - NUM_FRAME_DISCARDED - 1] = int(cls[-1])
            elif "Send" in cl:
                client_send[int(cls[0]) - NUM_FRAME_DISCARDED - 1] = int(cls[-1])
            elif "Recv" in cl:
                client_recv[int(cls[0]) - NUM_FRAME_DISCARDED - 1] = int(cls[-1])
            elif "Done" in cl:
                client_done[int(cls[0]) - NUM_FRAME_DISCARDED - 1] = int(cls[-1])

    with open(os.path.join(log_dir, server_log_file), "r") as server_log:
        server_lines = server_log.readlines()
        for sl in server_lines:
            if len(sl.strip()) > 0:
                sls = sl.split(", ")
                if sls[0] in ["#" + str(i + 1) for i in range(NUM_FRAME_DISCARDED)]:
                    continue
                idx = int(sls[0].replace("#", "")) - NUM_FRAME_DISCARDED - 1
                server_recv[idx] = float(sls[1].replace("time = ", "")) * 1000
                server_done[idx] = float(sls[2].replace("done = ", "")) * 1000
                server_pre[idx] = float(sls[3].replace("pre = ", "").replace(" ms", ""))
                server_detect[idx] = float(sls[4].replace("detect = ", "").replace(" ms", ""))
                server_repr[idx] = float(sls[5].replace("repr = ", "").replace(" ms", ""))
                server_infer[idx] = float(sls[6].replace("infer = ", "").replace(" ms", ""))
                server_post[idx] = float(sls[7].replace("post = ", "").replace(" ms", ""))
                server_wait[idx] = float(sls[8].replace("wait = ", "").replace(" ms", ""))
                server_total[idx] = server_done[idx] - server_recv[idx]

    num_frames = min(sum(client_done > 0), sum(server_done > 0))
    print(num_frames)
    client_gen = client_gen[:num_frames]
    client_send = client_send[:num_frames]
    client_recv = client_recv[:num_frames]
    client_done = client_done[:num_frames]
    server_recv = server_recv[:num_frames]
    server_done = server_done[:num_frames]
    server_pre = server_pre[:num_frames]
    server_detect = server_detect[:num_frames]
    server_repr = server_repr[:num_frames]
    server_infer = server_infer[:num_frames]
    server_post = server_post[:num_frames]
    server_total = server_total[:num_frames]

    server_wait = server_wait[1:num_frames]
    client_fps = [1000. / (client_done[i + 1] - client_done[i]) for i in range(num_frames - 1)]

    client_pre = [client_send[fid] - client_gen[fid] for fid in range(num_frames)]
    client_post = [client_done[fid] - client_recv[fid] for fid in range(num_frames)]
    net_uplink = [server_recv[fid] - client_send[fid] for fid in range(num_frames)]
    net_downlink = [client_recv[fid] - server_done[fid] for fid in range(num_frames)]
    net_total = [net_uplink[fid] + net_downlink[fid] for fid in range(num_frames)]
    frame_total = [client_done[fid] - client_gen[fid] for fid in range(num_frames)]
    stacked = np.stack([client_pre, net_uplink, server_pre, server_detect, server_repr, server_infer,
                        server_post, net_downlink, client_post])

    if plot_detailed_figures:
        """ Individual histograms
        """
        plot_histogram(client_pre, 'Time (ms)', 'Frame Count', os.path.join(log_dir, "client_pre.png"))
        plot_histogram(client_post, 'Time (ms)', 'Frame Count', os.path.join(log_dir, "client_post.png"))
        plot_histogram(server_pre, 'Time (ms)', 'Frame Count', os.path.join(log_dir, "server_pre.png"))
        plot_histogram(server_detect, 'Time (ms)', 'Frame Count', os.path.join(log_dir, "server_detect.png"))
        plot_histogram(server_repr, 'Time (ms)', 'Frame Count', os.path.join(log_dir, "server_repr.png"))
        plot_histogram(server_infer, 'Time (ms)', 'Frame Count', os.path.join(log_dir, "server_infer.png"))
        plot_histogram(server_post, 'Time (ms)', 'Frame Count', os.path.join(log_dir, "server_post.png"))
        plot_histogram(server_wait, 'Time (ms)', 'Frame Count', os.path.join(log_dir, "server_wait.png"))
        plot_histogram(server_total, 'Time (ms)', 'Frame Count', os.path.join(log_dir, "server_total.png"))
        plot_histogram(net_uplink, 'Latency (ms)', 'Frame Count', os.path.join(log_dir, "net_uplink.png"))
        plot_histogram(net_downlink, 'Latency (ms)', 'Frame Count', os.path.join(log_dir, "net_downlink.png"))
        plot_histogram(net_total, 'Latency (ms)', 'Frame Count', os.path.join(log_dir, "net_total.png"))
        plot_histogram(frame_total, 'Time (ms)', 'Frame Count', os.path.join(log_dir, "frame_total.png"))
        plot_histogram(client_fps, 'FPS', 'Frame Count', os.path.join(log_dir, "client_fps.png"), x_range=[0, 60])

        """ Per-frame individual timeline 
        """
        plot_individual_timeline(server_infer, 'Inference Time (ms)',
                                 os.path.join(log_dir, "server_infer_timeline.png"))
        plot_individual_timeline(net_uplink, 'Uplink Latency (ms)',
                                 os.path.join(log_dir, "net_uplink_timeline.png"))
        plot_individual_timeline(net_downlink, 'Downlink Latency (ms)',
                                 os.path.join(log_dir, "net_downlink_timeline.png"))

        """ Per-frame stacked timeline
        """
        labels = ["client_pre", "net_uplink", "server_pre", "server_detect", "server_repr", "server_infer",
                  "server_post", "net_downlink", "client_post"]
        fig, ax = plt.subplots()
        ax.set_xlabel('Frame Id')
        num_chosen = num_frames
        ax.set_xlim(0, num_chosen)
        ax.set_ylabel('Latency (ms)')
        ax.set_title('End-to-end Delay Distribution per Frame')
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
            np.mean(frame_total), np.median(frame_total), np.std(frame_total))
        ax.text(0.9, 0.85, agg_text, transform=ax.transAxes, bbox=dict(fc='white', ec='black', alpha=0.5))
        fig.savefig(os.path.join(log_dir, "frame_total_timeline_stacked.png"), dpi=600, bbox_inches="tight")
        plt.close(fig)

    return (np.mean(stacked, axis=1), np.median(stacked, axis=1), np.std(stacked, axis=1),
            np.percentile(stacked, 2.5, axis=1), np.percentile(stacked, 97.5, axis=1))


if __name__ == "__main__":
    BASEDIR = os.path.dirname(os.path.abspath(__file__))
    LOGDIR = os.path.join(BASEDIR, "logs/", "wifi_cloudlet_720p/")
    CLIENT_LOG = "Client-Timing.txt"
    SERVER_LOG = "Server-Log.txt"
    plot_trial(SERVER_LOG, CLIENT_LOG, LOGDIR, True)
