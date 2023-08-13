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

    fontname = "DejaVu Serif"
    fontsize = 18
    _ = fm.findfont(fm.FontProperties(family=fontname))
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = fontname

    for beta, title in (
        (0.5, "Underfitting"),
        (8.0, "Optimal"),
        (40.0, "Overfitting"),
    ):
        sns.set_style("ticks")
        g1 = np.exp(beta * ys1) / np.exp(beta * ys1).sum()
        g2 = np.exp(beta * ys2) / np.exp(beta * ys2).sum()
        plt.plot(xs1, g1, c="b")
        plt.plot(xs2, g2, c="r")
        plt.plot((0, 0), (g1.min(), 0.045), "--", c="g")
        plt.plot((1, 1), (g2.min(), 0.045), "--", c="g")
        plt.tick_params(axis="both", which="both", direction="in")
        plt.gca().axes.minorticks_on()
        plt.gca().axes.xaxis.set_ticklabels([])
        plt.gca().axes.yaxis.set_ticklabels([])
        plt.title(
            f"$\\beta = {beta}$" + f": {title}",
            fontsize=fontsize,
            fontname=fontname,
        )
        plt.xlabel("solutions", fontsize=fontsize, fontname=fontname)
        plt.ylabel("Gibbs posteriors", fontsize=fontsize, fontname=fontname)
        fname = osp.join(dirname, f"method_beta={beta}.pdf")
        plt.savefig(fname)
        plt.clf()
        plt.close()


def gibbs_posteriors_2d():
    dirname = osp.join("results", "plots", "paper")
    os.makedirs(dirname, exist_ok=True)

    xs, ys = np.meshgrid(np.linspace(-1, 2, 301), np.linspace(-1, 2, 301))
    v = np.stack((xs, ys))
    m1, s1 = np.array([[[-0.2]], [[+0.3]]]), np.array([[0.8, 0.4], [0.4, 1.2]])
    m2, s2 = np.array([[[1]], [[1.2]]]), np.array([[1.6, 0.1], [0.1, 0.3]])

    # s2 = (1, 2, 2) v = (n, n, 2)
    sub1 = (v - m1).reshape(2, -1)
    sub2 = (v - m2).reshape(2, -1)
    dst1 = (sub1 * (np.linalg.inv(s1) @ sub1)).sum(axis=0)
    dst2 = (sub2 * (np.linalg.inv(s2) @ sub2)).sum(axis=0)

    # lower normal part of Gaussian
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
        m1[0] + 0.7,
        m1[1] - 0.83,
        "$p(c \mid X^\prime)$",
        c=textcolor,
        fontsize=16,
        horizontalalignment="center",
        verticalalignment="center",
    )
    plt.text(
        m2[0] + 0.4,
        m2[1] - 0.58,
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
    plt.tight_layout()
    plt.savefig(fname)
    plt.clf()
    plt.close()


def beta_curve():
    dirname = osp.join("results", "plots", "paper")
    os.makedirs(dirname, exist_ok=True)

    or_betas = [0.5, 8.0, 40.0]
    or_pas = [0.2, 10.0, 0.01]
    betas = [0.47, 0.48, 0.5, 0.51, 5.5, 8.0, 15, 38.0, 40.0, 44.0]
    pas = [0.05, 0.1, 0.2, 0.25, 9.8, 10.0, 8, 0.2, 0.01, 0.0]
    cs = CubicSpline(betas, pas)
    xs = np.arange(0.0, 45.0, 0.1)

    # Text
    fontname = "DejaVu Serif"
    fontsize = 18
    _ = fm.findfont(fm.FontProperties(family=fontname))
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = fontname

    # Style
    sns.set_style("ticks")

    plt.figure(figsize=(18, 4))
    sns.set_style("ticks")
    plt.plot(xs, cs(xs), c="tab:orange")
    plt.xlim(0.0, 41.0)
    plt.xlabel("$\\beta$", fontsize=fontsize, fontname=fontname)
    plt.ylabel("Post. Agr.", fontsize=fontsize, fontname=fontname)
    textcolor = (0.35, 0.35, 0.3)
    plt.plot(or_betas, or_pas, "o", c=textcolor)
    offsets = ((1.35, 0.0), (0.0, -1.0), (-0.2, 0.8))
    for beta, pa, offset in zip(or_betas, or_pas, offsets):
        plt.text(
            beta + offset[0],
            pa + offset[1],
            f"$\\beta$ = {beta}",
            c=textcolor,
            fontsize=16,
            fontname=fontname,
            horizontalalignment="center",
            verticalalignment="center",
        )
    plt.tick_params(
        axis="both", which="both", direction="in", labelsize=fontsize - 2
    )
    plt.gca().spines["bottom"].set_color(textcolor)
    plt.gca().spines["top"].set_color(textcolor)
    plt.gca().spines["left"].set_color(textcolor)
    plt.gca().spines["right"].set_color(textcolor)

    plt.gca().axes.xaxis.set_ticklabels(np.arange(0, 45, 5))
    plt.gca().axes.yaxis.set_ticklabels([])
    # plt.title(f"$\\beta = {beta}$" + f": {title}" )

    fname = osp.join(dirname, f"gibbs_betas.pdf")
    plt.tight_layout()
    plt.savefig(fname)
    plt.clf()
    plt.close()


if __name__ == "__main__":
    # gibbs_posteriors_2d()
    gibbs_posteriors_2d()
