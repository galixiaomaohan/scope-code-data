# -*- coding: utf-8 -*-
# fig_llm_05_head_geometry.R -- Geometry of the operator-selection head rows.
#
# Claim (negative result, registered deliberately): the head's row geometry
# does NOT carry the 7-way curriculum partition -- the silhouette is negative.
# What it does carry is the legacy/added boundary: 156/161 base rows find their
# nearest neighbour inside the base block, while four added segments score
# 0.000; among the added 59 the 1-NN purity is 0.407 against chance 0.197.
# Panels: (a) PCA of W_head (220 x 1152), all rows; (b) UMAP of the same
#         matrix; (c) PCA recomputed on the 59 added rows only.
# Source: fig_head_geometry.csv (220 rows with PCA / UMAP coordinates).
# Discipline: the quantitative reading rests on the PCA panels (deterministic);
# UMAP depends on its seed. Identification is carried by the shared legend and
# the caption -- no text is placed on the point clouds.

source("scripts_r/pub_style_r.R")
source("scripts_r/pub_style_llm.R")
library(ggplot2)
library(patchwork)

d <- read.csv(here("fig_head_geometry.csv"))
lab_map <- c("Base (legacy 161)"  = "Base (legacy 161)",
             "Deep-chain ext."    = "Deep-chain ext. (17)",
             "Constr. curriculum" = "Constr. curriculum (7)",
             "Deep-chain family"  = "Deep-chain family (9)",
             "Subst. / eval."     = "Subst. / eval. (4)",
             "Routing"            = "Routing (9)",
             "Expansion"          = "Expansion (13)")
d$added <- as.logical(d$added_after_base)
d$seg <- factor(unname(lab_map[d$segment]), levels = names(SEG7))
stopifnot(!any(is.na(d$seg)), nrow(d) == 220, sum(d$added) == 59)
d <- d[order(d$added), ]                 # legacy base drawn first (underneath)
db <- d[!d$added, ]
da <- d[d$added, ]

# Equal-width y-tick boxes keep the two measured inter-panel gutters uniform
# (padding is invisible: y tick labels are right-aligned).
pad1 <- function(x) sprintf("%5.1f", x)
padi <- function(x) sprintf("%5d", x)
LEG  <- guide_legend(nrow = 1, byrow = TRUE,
                     override.aes = list(size = 1.6, alpha = 1))
SCOL <- scale_colour_manual(values = SEG7, breaks = levels(d$seg),
                            drop = FALSE)

## (a) PCA of W_head, all rows -----------------------------------------------
pa <- ggplot() +
  geom_point(data = db, aes(pc1_weight, pc2_weight, colour = seg),
             size = 0.7, alpha = 0.9) +
  # the added-rows layer carries the legend keys for its six segments (the
  # base layer only ever shows the single base segment)
  geom_point(data = da, aes(pc1_weight, pc2_weight, colour = seg),
             size = 0.95, show.legend = TRUE) +
  SCOL + scale_x_continuous(labels = function(x) sprintf("%5.1f", x)) +
  scale_y_continuous(labels = pad1) +
  labs(x = "PC1 (18% var)", y = "PC2 (7% var)", tag = "a") +
  theme_pub() + guides(colour = LEG)

## (b) UMAP of W_head, all rows ----------------------------------------------
pb <- ggplot() +
  geom_point(data = db, aes(umap1_weight, umap2_weight, colour = seg),
             size = 0.7, alpha = 0.9) +
  geom_point(data = da, aes(umap1_weight, umap2_weight, colour = seg),
             size = 0.95, show.legend = FALSE) +
  SCOL + scale_y_continuous(labels = padi) +
  labs(x = "UMAP-1", y = "UMAP-2", tag = "b") +
  theme_pub() + guides(colour = "none")

## (c) PCA of the 59 added rows only -----------------------------------------
pc <- ggplot(da, aes(pc1_addzoom, pc2_addzoom, colour = seg)) +
  geom_point(size = 0.95, show.legend = FALSE) +
  SCOL + scale_y_continuous(labels = pad1) +
  labs(x = "PC1 (29% var)", y = "PC2 (22% var)", tag = "c") +
  theme_pub() + guides(colour = "none")

fig <- (pa | pb | pc) +
  plot_layout(guides = "collect") +
  plot_annotation(
    tag_levels = "a",
    theme = theme(
      legend.position  = "bottom",
      legend.direction = "horizontal",
      legend.text      = element_text(size = 6.4, family = "Times New Roman"),
      legend.key.width = unit(7, "pt"),
      legend.spacing.x = unit(4, "pt")))

save_llm(fig, "output/fig_head_geometry", height_mm = 73,
         panel_ids = c("a", "b", "c"))
