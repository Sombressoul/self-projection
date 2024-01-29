import matplotlib.pyplot as plt


def plot_loss(
    loss: list[float],
):
    dpi = 100
    w_inches = 1200 / dpi
    h_inches = 1200 / dpi

    final_loss = loss[-1]
    min_loss = min(loss)
    max_loss = max(loss)
    epoch_count = len(loss)
    min_loss_idx = loss.index(min_loss)
    last_10_percent = loss[int(0.9 * epoch_count) :]
    last_10_mean = sum(last_10_percent) / len(last_10_percent)
    last_10_std = (
        sum([(x - last_10_mean) ** 2 for x in last_10_percent]) / len(last_10_percent)
    ) ** 0.5

    get_y = lambda x: (max_loss - min_loss) * x + min_loss

    plt.rcParams["figure.dpi"] = dpi
    plt.figure(figsize=(w_inches, h_inches))
    plt.plot(loss)
    plt.title("Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.annotate(
        f"Max Loss: {max_loss:.4f}",
        xy=(0, max_loss),
        xytext=(epoch_count / 2, get_y(1.0)),
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", lw=0.5),
    )
    plt.annotate(
        f"Final Loss: {final_loss:.4f}",
        xy=(epoch_count - 1, final_loss),
        xytext=(epoch_count / 2, get_y(0.8)),
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", lw=0.5),
    )
    plt.annotate(
        f"Min Loss: {min_loss:.4f}",
        xy=(min_loss_idx, min_loss),
        xytext=(epoch_count / 2, get_y(0.6)),
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", lw=0.5),
    )
    plt.figtext(
        0.5,
        0.01,
        f"Mean (last 10%): {last_10_mean:.4f}, Std (last 10%): {last_10_std:.4f}",
        ha="center",
        fontsize=10,
    )

    plt.show()
