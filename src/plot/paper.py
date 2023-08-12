import os
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def gibbs_posteriors():
    dirname = osp.join("results", "plots", "paper")
    os.makedirs(dirname, exist_ok=True)

    xs1 = np.linspace(-1, 2, 301)
    xs2 = np.linspace(-1, 2, 301)
    m1, s1 = 0, xs1.std()
    m2, s2 = 1, xs2.std()
    ys1 = np.exp(-0.5 * ((xs1 - m1) / s1) ** 2.0)
    ys2 = np.exp(-0.5 * ((xs2 - m2) / s2) ** 2.0)

    for beta, title in (
        (0.5, "Underfitting"),
        (8.0, "Optimal"),
        (40.0, "Overfitting"),
    ):
        sns.set_style("ticks")
        plt.style.use("science")
        g1 = np.exp(beta * ys1) / np.exp(beta * ys1).sum()
        g2 = np.exp(beta * ys2) / np.exp(beta * ys2).sum()
        plt.plot(xs1, g1, c="b")
        plt.plot(xs2, g2, c="r")
        plt.plot((0, 0), (g1.min(), 0.045), "--", c="g")
        plt.plot((1, 1), (g2.min(), 0.045), "--", c="g")
        plt.gca().axes.xaxis.set_ticklabels([])
        plt.gca().axes.yaxis.set_ticklabels([])
        plt.title(f"$\\beta = {beta}$" + f": {title}")
        plt.xlabel("solutions")
        plt.ylabel("Gibbs posteriors")
        fname = osp.join(dirname, f"method_beta={beta}.pdf")
        plt.savefig(fname)
        plt.clf()
        plt.close()


def gibbs_posteriors_2d():
    dirname = osp.join("results", "plots", "paper")
    os.makedirs(dirname, exist_ok=True)

    xs, ys = np.meshgrid(np.linspace(-1, 2, 301), np.linspace(-1, 2, 301))
    v = np.stack((xs, ys))
    m1, s1 = np.array([[[0]], [[0]]]), np.array([[0.8, 0.4], [0.4, 1.2]])
    m2, s2 = np.array([[[1]], [[1]]]), np.array([[1.6, 0.1], [0.1, 0.3]])

    # s2 = (1, 2, 2) v = (n, n, 2)
    sub1 = (v - m1).reshape(2, -1)
    sub2 = (v - m2).reshape(2, -1)
    dst1 = (sub1 * (np.linalg.inv(s1) @ sub1)).sum(axis=0)
    dst2 = (sub2 * (np.linalg.inv(s2) @ sub2)).sum(axis=0)

    # lower normal part of gaussian
    norm1 = np.sqrt((2 * np.pi) ** len(sub1) * np.linalg.det(s1))
    norm2 = np.sqrt((2 * np.pi) ** len(sub2) * np.linalg.det(s2))

    # Calculating Gaussian filter
    zs1 = (np.exp(-0.5 * dst1) / norm1).reshape(301, 301)
    zs2 = (np.exp(-0.5 * dst2) / norm2).reshape(301, 301)

    beta = 35.0

    g1 = np.exp(beta * zs1) / np.exp(beta * zs1).sum()
    g2 = np.exp(beta * zs2) / np.exp(beta * zs2).sum()

    plt.figure(figsize=(4, 4))
    sns.set_style("white")
    # plt.style.use("science")
    # plt.gca().axis("off")
    plt.axis("equal")
    plt.contour(xs, ys, g1, cmap="Blues")
    plt.contour(xs, ys, g2, cmap="Reds")
    textcolor = (0.35, 0.35, 0.3)
    plt.text(
        m1[0] + 1,
        m1[1] - 0.63,
        "$p(c \mid X^\prime)$",
        c=textcolor,
        fontsize=16,
        horizontalalignment="center",
        verticalalignment="center",
    )
    plt.text(
        m2[0] + 0,
        m2[1] + 0.68,
        "$p(c \mid X^{\prime\!\:\!\prime})$",
        c=textcolor,
        fontsize=16,
        horizontalalignment="center",
        verticalalignment="center",
    )
    plt.text(
        -0.9,
        1.85,
        "$\mathcal{C}$",
        c=textcolor,
        fontsize=16,
        horizontalalignment="center",
        verticalalignment="center",
    )
    plt.gca().spines["bottom"].set_color(textcolor)
    plt.gca().spines["top"].set_color(textcolor)
    plt.gca().spines["left"].set_color(textcolor)
    plt.gca().spines["right"].set_color(textcolor)

    plt.gca().axes.xaxis.set_ticklabels([])
    plt.gca().axes.yaxis.set_ticklabels([])
    # plt.title(f"$\\beta = {beta}$" + f": {title}" )

    fname = osp.join(dirname, f"gibbs2d_beta={beta}.pdf")
    plt.savefig(fname)
    plt.clf()
    plt.close()


if __name__ == "__main__":
    gibbs_posteriors_2d()
