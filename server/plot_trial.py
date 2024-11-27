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


def plot_trial(server_log_file, client_log_file, log_dir, plot_detailed_figures, client_ip=None):
    num_arrays = 9
    (client_gen, client_send, client_recv, client_done,
     server_recv, server_start, server_finish,
     client_payload, server_payload) = (
        tuple(np.squeeze(arr) for arr in np.split(np.zeros((num_arrays, NUM_TOTAL_FRAME_DEFAULT)), num_arrays, axis=0)))
    client_gen_csv_str = "id,gen\n"
    with open(os.path.join(log_dir, client_log_file), "r") as client_log:
        client_lines = client_log.readlines()
        for cl in client_lines:
            cls = cl.split()

            # Collect client gen time for measuring camera latency
            if "Gen" in cl and "Failed" not in cl:
                client_gen_csv_str += "{},{}\n".format(int(cls[0]), int(cls[-1]))

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

    with open(os.path.join(log_dir, "client_gen.csv"), "w") as csv_out:
        csv_out.write(client_gen_csv_str)

    with open(os.path.join(log_dir, server_log_file), "r") as server_log:
        server_lines = server_log.readlines()
        for sl in server_lines:
            if len(sl.strip()) > 0:
                sls = sl.split(",")

                if sls[3] in [str(i) for i in range(NUM_FRAME_DISCARDED + 1)]:
                    continue
                if client_ip is not None:
                    if sls[1].strip("('") != client_ip:
                        continue

                idx = int(sls[3].replace("#", "")) - NUM_FRAME_DISCARDED - 1
                if sls[0] == "Arrive":
                    server_recv[idx] = int(sls[-1])
                elif sls[0] == "Start":
                    server_start[idx] = int(sls[-1])
                elif sls[0] == "Finish":
                    server_finish[idx] = int(sls[-1])
                elif sls[0] == "ClientBytes":
                    client_payload[idx] = int(sls[-1])
                elif sls[0] == "ServerBytes":
                    server_payload[idx] = int(sls[-1])

    num_frames = min(sum(client_done > 0), sum(server_finish > 0))
    print(num_frames)
    client_gen = client_gen[:num_frames]
    client_send = client_send[:num_frames]
    client_recv = client_recv[:num_frames]
    client_done = client_done[:num_frames]
    server_recv = server_recv[:num_frames]
    server_start = server_start[:num_frames]
    server_finish = server_finish[:num_frames]
    client_payload = client_payload[:num_frames]
    server_payload = server_payload[:num_frames]

    client_fps = [1000. / (client_done[i + 1] - client_done[i]) for i in range(num_frames - 1)]

    client_pre = [client_send[fid] - client_gen[fid] for fid in range(num_frames)]
    client_post = [client_done[fid] - client_recv[fid] for fid in range(num_frames)]
    server_queue = [server_start[fid] - server_recv[fid] for fid in range(num_frames)]
    server_proc = [server_finish[fid] - server_start[fid] for fid in range(num_frames)]
    server_idle = [server_start[fid + 1] - server_start[fid] for fid in range(num_frames - 1)]
    net_uplink = [server_recv[fid] - client_send[fid] for fid in range(num_frames)]
    net_downlink = [client_recv[fid] - server_finish[fid] for fid in range(num_frames)]
    net_total = [net_uplink[fid] + net_downlink[fid] for fid in range(num_frames)]
    frame_total = [client_done[fid] - client_gen[fid] for fid in range(num_frames)]
    stacked = np.stack([client_pre, net_uplink, server_queue, server_proc,
                        net_downlink, client_post])

    if plot_detailed_figures:
        """ Individual histograms
        """
        plot_histogram(client_pre, 'Time (ms)', 'Frame Count', os.path.join(log_dir, "client_pre.png"))
        plot_histogram(client_post, 'Time (ms)', 'Frame Count', os.path.join(log_dir, "client_post.png"))
        plot_histogram(server_queue, 'Time (ms)', 'Frame Count', os.path.join(log_dir, "server_queue.png"))
        plot_histogram(server_proc, 'Time (ms)', 'Frame Count', os.path.join(log_dir, "server_proc.png"))
        plot_histogram(server_idle, 'Time (ms)', 'Frame Count', os.path.join(log_dir, "server_idle.png"))
        plot_histogram(net_uplink, 'Latency (ms)', 'Frame Count', os.path.join(log_dir, "net_uplink.png"))
        plot_histogram(net_downlink, 'Latency (ms)', 'Frame Count', os.path.join(log_dir, "net_downlink.png"))
        plot_histogram(net_total, 'Latency (ms)', 'Frame Count', os.path.join(log_dir, "net_total.png"))
        plot_histogram(frame_total, 'Time (ms)', 'Frame Count', os.path.join(log_dir, "frame_total.png"))
        plot_histogram(client_fps, 'FPS', 'Frame Count', os.path.join(log_dir, "client_fps.png"), x_range=[0, 60])
        plot_histogram(client_payload / 1000., 'Payload Size (KB)', 'Frame Count',
                       os.path.join(log_dir, "client_payload.png"))
        plot_histogram(server_payload / 1000., 'Payload Size (KB)', 'Frame Count',
                       os.path.join(log_dir, "server_payload.png"))

        """ Per-frame individual timeline 
        """
        plot_individual_timeline(server_proc, 'Server Processing Time (ms)',
                                 os.path.join(log_dir, "server_proc_timeline.png"))
        plot_individual_timeline(net_uplink, 'Uplink Latency (ms)',
                                 os.path.join(log_dir, "net_uplink_timeline.png"))
        plot_individual_timeline(net_downlink, 'Downlink Latency (ms)',
                                 os.path.join(log_dir, "net_downlink_timeline.png"))

        """ Per-frame stacked timeline
        """
        labels = ["client_pre", "net_uplink", "server_queue", "server_proc", "net_downlink", "client_post"]
        fig, ax = plt.subplots()
        ax.set_xlabel('Frame Id')
        num_chosen = num_frames
        ax.set_xlim(0, num_chosen)
        ax.set_ylabel('Latency (ms)')
        ax.set_title('End-to-end Latency Distribution on Each Frame')
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
    LOGDIR = os.path.join(BASEDIR, "logs/", "wifi_720p/")
    CLIENT_LOG = "Client.txt"

    user_ip = None
    # user_ip = "172.26.111.182"
    # user_ip = "172.26.50.76"
    # user_ip = "172.26.53.209"
    # user_ip = "172.26.111.147"

    SERVER_LOG = "Server.txt"
    plot_trial(SERVER_LOG, CLIENT_LOG, LOGDIR, True, user_ip)
