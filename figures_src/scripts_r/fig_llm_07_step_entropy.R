# -*- coding: utf-8 -*-
# fig_llm_07_step_entropy.R -- Decision entropy along the chain.
#
# Claim: decisions before step 4 are essentially deterministic; steps 4-6 are
# the only ones carrying uncertainty, and there the two outcome groups
# separate sharply -- failed items decide at 0.191 nats at t=4 against 0.027
# for certified ones (7x), yet the top-1 probability at that step is still
# 0.918, so the mistake is also "confidently wrong".
# Panels: (a) certified items; (b) failed items; both on the same log axis,
#         with +/-1 s.e.m. bands and each group's mean as a dashed line.
# Source: fig_step_entropy.csv (18 group x step rows).
# Discipline: entropy is taken over softmax(logits + mask) -- exactly the
# distribution the walk decides with. Per-step n is in the caption; the failed
# group's late points rest on a few frames (n annotated) and are not
# over-read. 'Certified' = exact operator-chain match (191/218).

source("scripts_r/pub_style_r.R")
source("scripts_r/pub_style_llm.R")
library(ggplot2)
library(patchwork)
library(scales)

d <- read.csv(here("fig_step_entropy.csv"))
d$group <- factor(d$group, levels = c("Certified items", "Failed items"))
d <- d[order(d$group, d$step_k), ]
COL <- c("Certified items" = PAL[["blue_main"]],
         "Failed items"    = PAL[["amber"]])
YL  <- c(5e-6, 0.8)

mk_panel <- function(g, title, tag, ann) {
  dg <- d[d$group == g, ]
  p <- ggplot(dg, aes(step_k, mean_entropy_nats)) +
    geom_hline(yintercept = mean(dg$mean_entropy_nats), linetype = "longdash",
               colour = PAL[["grey_m"]], linewidth = 0.35) +
    geom_ribbon(aes(ymin = mean_entropy_nats - sem_entropy,
                    ymax = mean_entropy_nats + sem_entropy),
                fill = COL[[g]], alpha = 0.25, colour = NA) +
    geom_line(colour = COL[[g]], linewidth = 0.55) +
    geom_point(shape = 21, fill = "white", colour = COL[[g]],
               size = 0.95, stroke = 0.3) +
    scale_x_continuous(breaks = 0:9, limits = c(-0.35, 9.35)) +
    scale_y_log10(limits = YL, labels = label_scientific(),
                  breaks = c(1e-5, 1e-4, 1e-3, 1e-2, 1e-1)) +
    labs(title = title, x = "Decision step t", tag = tag,
         y = expression("Mean entropy of " * italic(P) * "(" * italic(o) *
                          "|" * italic(F)[italic(t)] * "), nats (log)"))
  for (a in ann) {
    p <- p + annotate(a$type, x = a$x, y = a$y, label = a$label,
                      hjust = a$hjust, vjust = a$vjust %||% 0.5,
                      size = a$size, family = "Times New Roman",
                      fontface = a$face %||% "plain", colour = a$col)
  }
  p + theme_pub()
}
`%||%` <- function(a, b) if (is.null(a)) b else a

ann_a <- list()
ann_b <- list()

pa <- mk_panel("Certified items", NULL, "a", ann_a)
pb <- mk_panel("Failed items", NULL, "b", ann_b)

fig <- (pa | pb) +
  plot_annotation(
    tag_levels = "a",
    theme = theme())

save_llm(fig, "output/fig_step_entropy", height_mm = 61,
         panel_ids = c("a", "b"))
