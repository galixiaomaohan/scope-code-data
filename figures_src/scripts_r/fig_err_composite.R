# fig_err_composite.R -- Composition, confidence and entropy of terminal-step
# errors (paper Fig. 4, one row of three panels, 183 x 60 mm).
# v1 (2026-10-02): consolidates the former fig_err_confidence and
# fig_step_entropy into the main-text composite requested for the revision.
#   (a) confusion class of the first errors, two configurations
#       (gen36 in-series pairing: baseline 101 / final G-S 27);
#   (b) ECDF of top-1 probability at the first erroneous step, 0.9 zone shaded,
#       confidently-wrong 55 -> 12;
#   (c) per-step decision entropy, certified vs failed, first divergence 7x.
# Data: ../data/processed/figures/fig_err_confidence.csv, .../fig_step_entropy.csv.
# Counts verified against GEN36_ERRANALYSIS.json (zero-class 36 -> 0; 31/31).
# v2 (2026-10-03): hairline axes 0.3 mm (matched to the matplotlib track's
# 0.8 pt reference); canvas 183 x 62 -> 183 x 78 mm so the three panels are
# no longer squat; data lines thinned accordingly.
library(ggplot2)
library(patchwork)
source("scripts_r/pub_style_r.R")

theme_big <- function() {
  theme_pub(base_size = 10) +
    theme(
      axis.line         = element_line(linewidth = 0.3, colour = "black"),
      axis.ticks        = element_line(linewidth = 0.3, colour = "black"),
      axis.ticks.length = unit(2.5, "pt"),
      axis.title        = element_text(size = 10, face = "bold", colour = "black"),
      axis.text         = element_text(size = 9, colour = "black"),
      plot.tag          = element_text(size = 12, face = "bold",
                                       family = "Times New Roman")
    )
}

## (a) error composition (dumbbell dot plot) ----------------------------------
labs3 <- c("Gold square,\npredicted other", "Gold other,\npredicted square",
           "Zero-class\ninvolved")
dd <- data.frame(
  cls = rep(labs3, times = 2),
  cfg = rep(c("Baseline (101 errors)", "Final (27 errors)"), each = 3),
  y   = rep(3:1, times = 2),
  n   = c(41, 24, 36, 9, 18, 0))
dw <- reshape(dd, idvar = c("cls", "y"), timevar = "cfg",
              direction = "wide", sep = "__")
names(dw)[names(dw) == "n__Baseline (101 errors)"] <- "n_base"
names(dw)[names(dw) == "n__Final (27 errors)"] <- "n_fin"

p_a <- ggplot(dd, aes(n, y)) +
  geom_segment(data = dw,
               aes(x = n_base, xend = n_fin, y = y, yend = y),
               colour = "#C9C9C9", linewidth = 0.3) +
  geom_point(data = subset(dd, cfg == "Baseline (101 errors)"),
             size = 3.0, colour = PAL$grey_m) +
  geom_point(data = subset(dd, cfg == "Final (27 errors)"),
             size = 3.0, colour = PAL$blue_main) +
  scale_x_continuous(limits = c(-0.5, 54), breaks = c(0, 10, 20, 30, 40),
                     expand = c(0, 0)) +
  scale_y_continuous(breaks = 3:1, labels = labs3,
                     limits = c(0.55, 3.85), expand = c(0, 0)) +
  labs(x = "First errors (count)", y = NULL, colour = NULL) +
  theme_big() +
  theme(axis.text.y = element_text(size = 8.2, colour = "black"),
        legend.position = "none")

## (b) confidence ECDF --------------------------------------------------------
d <- read.csv("../data/processed/figures/fig_err_confidence.csv")
d$config <- factor(d$config, levels = c("baseline", "final"),
                   labels = c("Baseline", "Final (state shift)"))
COL <- c("Baseline" = PAL$grey_m, "Final (state shift)" = PAL$blue_main)

ecdf_dat <- do.call(rbind, lapply(split(d, d$config), function(g) {
  data.frame(config = g$config[1], p_max = sort(g$p_max),
             ecdf = seq_len(nrow(g)) / nrow(g))
}))

p_b <- ggplot(ecdf_dat, aes(p_max, ecdf, colour = config)) +
  geom_rect(xmin = 0.9, xmax = 1.0, ymin = 0, ymax = 1,
            fill = "#F5D9A0", alpha = 0.30, colour = NA) +
  geom_step(linewidth = 0.4) +
  geom_vline(xintercept = 0.9, linetype = "longdash", colour = PAL$grey_m,
             linewidth = 0.3) +
  annotate("point", x = 0.9, y = 55 / 101, colour = PAL$grey_d, size = 2.2) +
  annotate("point", x = 0.9, y = 12 / 27, colour = PAL$blue_main, size = 2.2) +
  coord_cartesian(xlim = c(0.3, 1.005), ylim = c(0, 1)) +
  scale_x_continuous(breaks = seq(0.3, 1.0, 0.1)) +
  scale_colour_manual(values = COL) +
  guides(colour = guide_legend(override.aes = list(linewidth = 0.6))) +
  labs(x = expression(bold("Top-1 probability at first error  ") *
                        bolditalic(p[max])),
       y = "Cumulative fraction of errors") +
  theme_big()

## (c) per-step decision entropy ----------------------------------------------
e <- read.csv("../data/processed/figures/fig_step_entropy.csv")
e$grp <- factor(e$group,
                levels = c("Certified items", "Failed items"),
                labels = c("Certified", "Failed"))
COL_E <- c("Certified" = PAL$blue_main, "Failed" = PAL$red_s)

p_c <- ggplot(e, aes(step_k, mean_entropy_nats, colour = grp,
                     fill = grp, group = grp)) +
  geom_ribbon(aes(ymin = pmax(mean_entropy_nats - sem_entropy, 2e-6),
                  ymax = mean_entropy_nats + sem_entropy),
              alpha = 0.16, colour = NA) +
  geom_line(linewidth = 0.4) +
  geom_point(size = 2.1) +
  scale_y_log10(limits = c(5e-6, 2.5),
                breaks = c(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1),
                labels = parse(text = c("10^-5", "10^-4", "10^-3",
                                        "10^-2", "10^-1", "10^0"))) +
  scale_x_continuous(breaks = c(0, 2, 4, 6, 8), limits = c(-0.15, 9.3),
                     expand = c(0, 0)) +
  scale_colour_manual(values = COL_E, name = NULL) +
  scale_fill_manual(values = COL_E, guide = "none") +
  labs(x = "Decision step t", y = "Decision entropy (nats)") +
  theme_big() +
  theme(legend.position = c(0.99, 0.99), legend.justification = c(1, 1),
        legend.direction = "horizontal",
        legend.text = element_text(size = 8.5, colour = "black"),
        legend.key.height = unit(8, "pt"),
        legend.spacing.x = unit(2, "pt"),
        legend.background = element_rect(fill = alpha("white", 0.75),
                                         colour = NA))

fig <- (p_a | p_b | p_c) +
  plot_layout(guides = "collect") +
  plot_annotation(tag_levels = "a") &
  theme(legend.position = "bottom",
        legend.text = element_text(size = 8.5, family = "Times New Roman",
                                   colour = "black"),
        legend.key.height = unit(8, "pt"))
save_pub_r(fig, "output/fig_err_composite", width_mm = 183, height_mm = 71,
           panel_ids = c("a", "b", "c"),
           exemptions = list(list(
             panels = c("a", "b", "c"), checks = "horizontal-gutter",
             reason = paste("y tick label widths differ by design (family names /",
                            "decimals / 1e-notation); gutter spread is the residue,",
                            "visually negligible"))))
