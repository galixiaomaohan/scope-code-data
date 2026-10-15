# -*- coding: utf-8 -*-
# fig_llm_08_loss_landscape.R -- Filter-normalised loss landscape around the
# final checkpoint.
#
# Claim: the tail-decision planning loss sits in a shallow but well-defined
# basin -- loss(theta*) = 0.351 nats, grid minimum 0.235, grid corners 5.34
# nats (15.2x) -- i.e. a broad flat valley rather than a sharp spike,
# consistent with the very low decision entropy of fig_step_entropy.
# Panels: (a) filled contour map of the 51x51 grid with the optimum marked;
#         (b) the two axis cross-sections through theta* (this REPLACES the
#         original 3-D surface, which only redrew the same grid: a section
#         carries a distinct reading -- how fast the loss rises with
#         distance).
# Source: fig_loss_landscape.csv (2601 = 51x51 grid points).
# Discipline: this is the tail-decision planning loss at step k=4 (teacher
# forced, gold prefix), NOT the full training objective; delta = 0.20 was set
# by a 1-D calibration scan -- the shape is the claim, not absolute widths.

source("scripts_r/pub_style_r.R")
source("scripts_r/pub_style_llm.R")
library(ggplot2)
library(patchwork)
library(viridisLite)

d <- read.csv(here("fig_loss_landscape.csv"))
stopifnot(nrow(d) == 2601, length(unique(d$alpha_d1)) == 51,
          length(unique(d$beta_d2)) == 51)

LSTAR <- 0.3513988   # loss(theta*), from the run record
LMIN  <- 0.2348931
LMAX  <- 5.3397575

s1 <- d[abs(d$beta_d2) < 1e-9, ]    # section along d1 (beta = 0)
s2 <- d[abs(d$alpha_d1) < 1e-9, ]   # section along d2 (alpha = 0)
s1 <- s1[order(s1$alpha_d1), ]
s2 <- s2[order(s2$beta_d2), ]
yhi <- 2.75                          # headroom for the corner labels

## (a) filled contour map ----------------------------------------------------
pa <- ggplot(d, aes(alpha_d1, beta_d2)) +
  geom_raster(aes(fill = loss_nats)) +
  geom_contour(aes(z = loss_nats), colour = "white", linewidth = 0.18,
               alpha = 0.7,
               breaks = c(0.4, 0.5, 0.75, 1, 1.5, 2, 3, 4)) +
  annotate("point", x = 0, y = 0, shape = 8, size = 3.2, colour = "white") +
  annotate("point", x = 0, y = 0, shape = 8, size = 1.8,
           colour = PAL[["ink"]]) +
  scale_fill_viridis_c(option = "viridis",
                       name = "loss (nats)",
                       guide = guide_colourbar(direction = "horizontal",
                                               barwidth = unit(56, "pt"),
                                               barheight = unit(5.5, "pt"),
                                               title.position = "top",
                                               title.hjust = 0.5)) +
  labs(x = expression(alpha ~ "along " ~ italic(d)[1]),
       y = expression(beta ~ "along " ~ italic(d)[2]), tag = "a") +
  theme_pub()

## (b) cross-sections through the optimum -------------------------------------
sec <- rbind(data.frame(x = s1$alpha_d1, loss = s1$loss_nats,
                        dir = "along d1"),
             data.frame(x = s2$beta_d2, loss = s2$loss_nats,
                        dir = "along d2"))
sec$dir <- factor(sec$dir, levels = c("along d1", "along d2"))
SCOL <- c("along d1" = PAL[["blue_main"]], "along d2" = PAL[["amber"]])

pb <- ggplot(sec, aes(x, loss, colour = dir)) +
  geom_hline(yintercept = LSTAR, linetype = "longdash",
             colour = PAL[["grey_m"]], linewidth = 0.35) +
  geom_line(linewidth = 0.5) +
  annotate("point", x = 0, y = LSTAR, shape = 21, fill = "white",
           colour = PAL[["ink"]], size = 1.1, stroke = 0.3) +
  # direct labels live in the empty top-left corner (curves never exceed
  # ~1.8 nats on the left half); colours tie them to their series.
  annotate("text", x = -0.19, y = 2.48, hjust = 0, size = 2.3,
           family = "Times New Roman", fontface = "bold",
           colour = PAL[["blue_main"]], label = "along d1") +
  annotate("text", x = -0.19, y = 2.18, hjust = 0, size = 2.3,
           family = "Times New Roman", fontface = "bold",
           colour = PAL[["amber"]], label = "along d2") +
  scale_colour_manual(values = SCOL) +
  scale_x_continuous(breaks = c(-0.2, -0.1, 0, 0.1, 0.2)) +
  coord_cartesian(ylim = c(0, yhi)) +
  labs(x = expression("perturbation " * delta * "  (-0.20 .. 0.20)"),
       y = "Loss (nats)", tag = "b", colour = NULL) +
  theme_pub() + guides(colour = "none")

fig <- (pa | pb) +
  plot_layout(guides = "collect") +
  plot_annotation(
    tag_levels = "a",
    theme = theme(legend.position = "bottom"))

save_llm(fig, "output/fig_loss_landscape", height_mm = 86,
         panel_ids = c("a", "b"))
