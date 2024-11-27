import os
import numpy as np
import matplotlib.pyplot as plt
from plot_trial import *

BASEDIR = os.path.dirname(os.path.abspath(__file__))


def plot_comparison_by_group(data_to_compare, config_names, group_xticks, title, output_file):
    labels = ["client_pre", "net_uplink", "server_queue", "server_proc",
              "net_downlink", "client_post"]
    # colors = ["cyan", "yellow", "green", "red", "blue", "magenta", "lime"]
    y_data = np.stack([data[0] for data in data_to_compare]).T
    # error_bar_top = y_data.copy()
    # y_error_lower = np.stack([data[0] - data[3] for data in data_to_compare]).T
    # y_error_upper = np.stack([data[4] - data[0] for data in data_to_compare]).T

    fig, ax = plt.subplots()
    ax.set_ylabel('Time (ms)')
    ax.set_title(title)

    num_groups = len(group_xticks)
    group_size = len(config_names) // num_groups
    scaling = 2.
    offset = 0.8 / group_size
    width = (offset - 0.01) * scaling
    x = np.repeat(np.arange(num_groups), group_size)
    offsets = np.array([-0.32, -0.16, 0., 0.16, 0.32] * num_groups)  # TODO: Use np.linspace
    x_grouped = (x + offsets) * scaling
    ax.set_xticks(x_grouped, config_names, rotation=25, ha='right')
    bottom = np.zeros(len(config_names))
    for i in range(len(labels)):
        p = ax.bar(x_grouped, y_data[i], label=labels[i], capsize=3, width=width,
                   bottom=bottom)  # ,yerr=[y_error_lower[i], y_error_upper[i]])  # , color=colors[i])
        bottom += y_data[i]
        # error_bar_top[i] = bottom + y_error_upper[i]
    for x_pos in range(len(x_grouped)):
        ax.text(x_grouped[x_pos], bottom[x_pos] + 2., round(bottom[x_pos]), ha='center',
                fontsize=10)  # np.max(error_bar_top[:, x_pos]) + 1.

    # Reorder the labels in the legend
    handles, lbs = ax.get_legend_handles_labels()
    handles.reverse()
    lbs.reverse()
    ax.legend(handles, lbs, loc="upper right")

    sec = ax.secondary_xaxis(location=0)
    sec.set_xticks((np.arange(len(group_xticks))) * scaling,
                   labels=["\n\n\n\n" + xtick for xtick in group_xticks])
    sec.tick_params('x', length=0)

    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(data_to_compare, config_names, title, output_file):
    labels = ["client_pre", "net_uplink", "server_queue", "server_proc",
              "net_downlink", "client_post"]
    y_data = np.stack([data[0] for data in data_to_compare]).T

    x = np.arange(len(config_names))
    fig, ax = plt.subplots()
    ax.set_xticks(x, config_names)
    ax.set_ylabel('Time (ms)')
    ax.set_title(title)

    bottom = np.zeros(len(config_names))
    for i in range(len(labels)):
        p = ax.bar(x, y_data[i], label=labels[i], capsize=3, width=0.5, bottom=bottom)
        bottom += y_data[i]
    for x_pos in range(len(config_names)):
        ax.text(x_pos, bottom[x_pos] + 2., round(bottom[x_pos]), ha='center',
                fontsize=10)

    ax.set_ylim(0, np.max(bottom) + 40.)

    # Reorder the labels in the legend
    handles, lbs = ax.get_legend_handles_labels()
    handles.reverse()
    lbs.reverse()
    ax.legend(handles, lbs, loc="upper left")
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_by_custom_group(data_to_compare, config_names, group_xticks, title, output_file):
    labels = ["client_pre", "net_uplink", "server_queue", "server_proc",
              "net_downlink", "client_post"]
    y_data = np.stack([data[0] for data in data_to_compare]).T

    fig, ax = plt.subplots()
    ax.set_ylabel('Time (ms)')
    ax.set_title(title)

    scaling = 2.
    x_grouped = np.array([0., 0.32, 0.48, 0.8, 0.96, 1.12, 1.44, 1.6, 1.76, 1.92]) * scaling
    offset = 0.16
    width = (offset - 0.01) * scaling
    ax.set_xticks(x_grouped, config_names)
    bottom = np.zeros(len(config_names))
    for i in range(len(labels)):
        p = ax.bar(x_grouped, y_data[i], label=labels[i], capsize=3, width=width, bottom=bottom)
        bottom += y_data[i]
    for x_pos in range(len(x_grouped)):
        ax.text(x_grouped[x_pos], bottom[x_pos] + 2., round(bottom[x_pos]), ha='center', fontsize=10)

    # Reorder the labels in the legend
    handles, lbs = ax.get_legend_handles_labels()
    handles.reverse()
    lbs.reverse()
    ax.legend(handles, lbs, loc="upper left")

    sec = ax.secondary_xaxis(location=0)
    sec.set_xticks(np.array([0., 0.4, 0.96, 1.68]) * scaling,
                   labels=["\n" + xtick for xtick in group_xticks])
    sec.tick_params('x', length=0)

    ax.set_ylim(0, np.max(bottom) + 120.)

    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # 480p vs 720p
    data_folders = ["wifi_720p/", "tmobile_4g_720p/", "tmobile_5g_720p/", "cbrs_4g_720p/", "cbrs_5g_720p/",
                    "wifi_480p/", "tmobile_4g_480p/", "tmobile_5g_480p/", "cbrs_4g_480p/", "cbrs_5g_480p/"]
    data_results = [plot_trial("Server.txt", "Client.txt",
                               os.path.join(BASEDIR, "logs/", data_folder), False)
                    for data_folder in data_folders]
    trial_names = ["CMU Wi-Fi", "T-Mobile 4G", "T-Mobile 5G", "CBRS 4G", "CBRS 5G",
                   "CMU Wi-Fi", "T-Mobile 4G", "T-Mobile 5G", "CBRS 4G", "CBRS 5G"]
    group_names = ["720p", "480p"]
    plot_comparison_by_group(data_results, trial_names, group_names, "OpenFace MTPL Comparison",
                             os.path.join(BASEDIR, "logs/", "breakdown_comparison.png"))

    # token count
    data_folders = ["wifi_720p/", "wifi_720p_token_2/", "wifi_720p_token_3/", "wifi_720p_token_4/"]
    data_results = [plot_trial("Server.txt", "Client.txt",
                               os.path.join(BASEDIR, "logs/", data_folder), False)
                    for data_folder in data_folders]
    trial_names = ["1 token", "2 tokens", "3 tokens", "4 tokens"]
    plot_comparison(data_results, trial_names, "OpenFace MTPL Comparison (CMU Wi-Fi 720p)",
                    os.path.join(BASEDIR, "logs/", "token_comparison.png"))

    # client count
    data_result_1 = plot_trial("Server.txt", "Client.txt",
                               os.path.join(BASEDIR, "logs/", "wifi_720p/"), False)
    data_result_2 = [plot_trial("Server.txt", "Client1.txt",
                               os.path.join(BASEDIR, "logs/", "wifi_720p_2_clients/"),
                                False, "172.26.111.182"),
                     plot_trial("Server.txt", "Client2.txt",
                                os.path.join(BASEDIR, "logs/", "wifi_720p_2_clients/"),
                                False, "172.26.111.147")]
    data_result_3 = [plot_trial("Server.txt", "Client1.txt",
                               os.path.join(BASEDIR, "logs/", "wifi_720p_3_clients/"),
                                False, "172.26.111.182"),
                     plot_trial("Server.txt", "Client2.txt",
                                os.path.join(BASEDIR, "logs/", "wifi_720p_3_clients/"),
                                False, "172.26.50.76"),
                     plot_trial("Server.txt", "Client3.txt",
                                os.path.join(BASEDIR, "logs/", "wifi_720p_3_clients/"),
                                False, "172.26.111.147")]
    data_result_4 = [plot_trial("Server.txt", "Client1.txt",
                               os.path.join(BASEDIR, "logs/", "wifi_720p_4_clients/"),
                                False, "172.26.111.182"),
                     plot_trial("Server.txt", "Client2.txt",
                                os.path.join(BASEDIR, "logs/", "wifi_720p_4_clients/"),
                                False, "172.26.50.76"),
                     plot_trial("Server.txt", "Client3.txt",
                                os.path.join(BASEDIR, "logs/", "wifi_720p_4_clients/"),
                                False, "172.26.53.209"),
                     plot_trial("Server.txt", "Client4.txt",
                                os.path.join(BASEDIR, "logs/", "wifi_720p_4_clients/"),
                                False, "172.26.111.147")]
    data_results = [data_result_1, data_result_2[0], data_result_2[1], data_result_3[0], data_result_3[1],
                    data_result_3[2], data_result_4[0], data_result_4[1], data_result_4[2], data_result_4[3]]
    trial_names = ["A", "A", "B", "A", "B", "C", "A", "B", "C", "D"]
    group_names = ["1 client", "2 clients", "3 clients", "4 clients"]
    plot_comparison_by_custom_group(data_results, trial_names, group_names,
                                    "OpenFace MTPL Comparison (CMU Wi-Fi 720p)",
                                    os.path.join(BASEDIR, "logs/", "client_comparison.png"))
