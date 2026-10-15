# -*- coding: utf-8 -*-
# fig_llm_02_err_confidence.R -- Error-confidence profile, two configurations.
#
# Claim: the final configuration does not err more humbly -- the ECDF shape is
# nearly unchanged -- it simply errs less often (confidently-wrong 55 -> 12).
# Panels: (a) hero: ECDF of p_max at the first erroneous step, two configs,
#         0.9 marker with counts; (b)+(c) absolute histograms with rug.
# Source: fig_err_confidence.csv (128 rows: 101 baseline + 27 final errors).
# Discipline: gen36 in-series pairing (baseline GEN35 vs final G-S), the same
# source as Section 8; the s1 re-test 13/27 reading coexists and is NOT drawn.

source("scripts_r/pub_style_r.R")
source("scripts_r/pub_style_llm.R")
library(ggplot2)
library(patchwork)
library(scales)

d <- read.csv(here("fig_err_confidence.csv"))
d$config <- factor(d$config, levels = c("baseline", "final"),
                   labels = c("baseline", "final (state shift)"))
COL <- c("baseline" = PAL[["blue_main"]],
         "final (state shift)" = PAL[["amber"]])

ecdf_dat <- do.call(rbind, lapply(split(d, d$config), function(g) {
  data.frame(config = g$config[1], p_max = sort(g$p_max),
             ecdf = seq_len(nrow(g)) / nrow(g))
}))

## (a) ECDF ------------------------------------------------------------------
# All annotation lives in the empty top-left band (the curves only enter it at
# the far right): two short lines, no per-config floating counts, legend
# collected at the figure bottom so nothing sits on the step functions.
pa <- ggplot(ecdf_dat, aes(p_max, ecdf, colour = config)) +
  geom_rect(xmin = 0.9, xmax = 1.0, ymin = 0, ymax = 1,
            fill = "#F5D9A0", alpha = 0.25, colour = NA) +
  geom_step(linewidth = 0.6) +
  geom_vline(xintercept = 0.9, linetype = "longdash", colour = PAL[["grey_m"]],
             linewidth = 0.35) +
  annotate("point", x = 0.9, y = 55 / 101, colour = PAL[["amber"]], size = 1.4) +
  annotate("point", x = 0.9, y = 12 / 27, colour = PAL[["blue_main"]],
           size = 1.4) +
  coord_cartesian(xlim = c(0.3, 1.005), ylim = c(0, 1)) +
  scale_x_continuous(breaks = seq(0.3, 1.0, 0.1)) +
  scale_colour_manual(values = COL,
                      guide = guide_legend(nrow = 1, byrow = TRUE,
                                           override.aes = list(linewidth = 1.1))) +
  labs(x = expression("Top-1 probability at the first erroneous step  " *
                        italic(p[max])),
       y = "Cumulative fraction of errors", tag = "a", colour = NULL) +
  theme_pub()

## (b)(c) histograms ---------------------------------------------------------
brk <- seq(0.3, 1.0, 0.05)
levs <- levels(d$config)
mk_hist <- function(cfg, n_err, n_conf, n_cert) {
  g <- d[d$config == levs[cfg + 1], ]
  key <- levs[cfg + 1]
  h <- hist(g$p_max, breaks = brk, plot = FALSE)
  hd <- data.frame(mid = h$mids, count = h$counts, config = key)
  rug <- data.frame(x = g$p_max)
  ggplot(hd, aes(mid, count)) +
    geom_col(aes(fill = config), width = 0.046, colour = "white",
             linewidth = 0.15) +
    geom_vline(xintercept = 0.9, linetype = "longdash",
               colour = PAL[["grey_m"]], linewidth = 0.3) +
    geom_rect(xmin = 0.9, xmax = 1.0, ymin = 0, ymax = max(hd$count) * 1.25,
              fill = "#F5D9A0", alpha = 0.22, colour = NA) +
    # rug as short strokes: ggplot's glyph shapes (e.g. 62) are emitted as
    # text runs and would sit below the 5 pt glyph floor in the PDF audit
    geom_segment(data = rug, aes(x = x, xend = x, y = -0.55, yend = 0),
                 linewidth = 0.18,
                 colour = if (cfg == 0) PAL[["blue_main"]] else PAL[["amber"]]) +
    coord_cartesian(xlim = c(0.3, 1.005), ylim = c(-0.6, max(hd$count) * 1.32)) +
    scale_fill_manual(values = COL, guide = "none") +
    labs(x = NULL, y = "Erroneous items (count)", tag = if (cfg == 0) "b" else "c") +
    theme_pub()
}
pb <- mk_hist(0, 101, 55, "117/218")
pc <- mk_hist(1, 27, 12, "191/218")

fig <- (pa | pb | pc) +
  plot_layout(guides = "collect") +
  plot_annotation(
    tag_levels = "a",
    theme = theme(legend.position = "bottom",
                  legend.text = element_text(size = 6.4,
                                             family = "Times New Roman")))

save_llm(fig, "output/fig_err_confidence", height_mm = 70,
         panel_ids = c("a", "b", "c"))
