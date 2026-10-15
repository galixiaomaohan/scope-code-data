# -*- coding: utf-8 -*-
# fig_llm_06_hidden_trajectory.R -- Geometry of the decision-frame hidden
# states h_t (1152-d, the exact input of the operator-selection head).
#
# Claim: h_t is organised first by problem family -- the three families
# separate almost perfectly (1-NN purity 0.999 in the embedding; chance 0.659)
# -- and then by position inside the chain: consecutive frames of one item are
# far more similar than random pairs (cosine 0.607 vs 0.154). A chain walks a
# family-specific route through state space rather than staying put.
# Panels: (a) all 1552 frames, coloured by family; (b) the 80 sampled chains
#         as step-ordered polylines (open = first frame, solid = last);
#         (c) all frames, coloured by step index.
# Source: fig_hidden_trajectory.csv (1552 frames x 218 items).
# Discipline: the family purity is a statement about the 2-D embedding; the
# cosine comparison lives in the full 1152-d space. Nothing is placed on the
# point clouds -- keys sit in the titles, numbers in the caption.

source("scripts_r/pub_style_r.R")
source("scripts_r/pub_style_llm.R")
library(ggplot2)
library(patchwork)
library(viridisLite)

d <- read.csv(here("fig_hidden_trajectory.csv"))
fam_map <- c(gen32_ind_nest    = "Nested induction",
             gen32_ind_tower   = "Lemma tower",
             gen32_ineq_cascade = "Sum-of-squares cascade")
d$family <- factor(unname(fam_map[d$family]), levels = names(FAM3))
stopifnot(!any(is.na(d$family)), nrow(d) == 1552, length(unique(d$ri)) == 218)
d$item_exact_hit <- as.logical(d$item_exact_hit)

track <- d[d$in_trajectory_sample == 1, ]
stopifnot(length(unique(track$ri)) == 80)
track <- track[order(track$ri, track$step_k), ]
firsts <- track[!duplicated(track$ri), ]
lasts  <- track[!duplicated(track$ri, fromLast = TRUE), ]

KMAX <- 9
xr <- range(d$umap1) + c(-1.5, 1.5)
yr <- range(d$umap2) + c(-1.5, 1.5)
STOP <- function(p) p +
  scale_x_continuous(limits = xr) + scale_y_continuous(limits = yr)

LEG <- guide_legend(nrow = 1, byrow = TRUE,
                    override.aes = list(size = 1.6, alpha = 1))
SCOL <- scale_colour_manual(values = FAM3, breaks = names(FAM3),
                            drop = FALSE)

## (a) all frames, by family --------------------------------------------------
pa <- STOP(ggplot(d, aes(umap1, umap2, colour = family)) +
  geom_point(size = 0.65, alpha = 0.85) +
  SCOL +
  labs(x = "UMAP-1", y = "UMAP-2", tag = "a", colour = NULL) +
  theme_pub() + guides(colour = LEG))

## (b) 80 sampled chains as polylines ----------------------------------------
pb <- STOP(ggplot(track, aes(umap1, umap2)) +
  geom_path(aes(group = ri, colour = family), linewidth = 0.22, alpha = 0.5) +
  geom_point(data = firsts, aes(colour = family), shape = 21, fill = "white",
             size = 0.75, stroke = 0.3) +
  geom_point(data = lasts, aes(colour = family, fill = family), shape = 24,
             size = 0.8, stroke = 0.25) +
  SCOL + scale_fill_manual(values = FAM3, guide = "none") +
  labs(x = "UMAP-1", y = "UMAP-2", tag = "b", colour = NULL) +
  theme_pub() + guides(colour = "none"))

## (c) all frames, by step index ---------------------------------------------
pc <- STOP(ggplot(d, aes(umap1, umap2, colour = step_k)) +
  geom_point(size = 0.65, alpha = 0.85) +
  scale_colour_viridis_c(option = "viridis", limits = c(0, KMAX),
                         breaks = c(0, 2, 4, 6, 8), name = "decision step t",
                         guide = guide_colourbar(direction = "horizontal",
                                                 barwidth = unit(50, "pt"),
                                                 barheight = unit(5.5, "pt"),
                                                 title.position = "top",
                                                 title.hjust = 0.5)) +
  labs(x = "UMAP-1", y = "UMAP-2", tag = "c") +
  theme_pub())

fig <- (pa | pb | pc) +
  plot_layout(guides = "collect") +
  plot_annotation(
    tag_levels = "a",
    theme = theme(
      legend.position  = "bottom",
      legend.text      = element_text(size = 6.4, family = "Times New Roman"),
      legend.title     = element_text(size = 6.4, family = "Times New Roman"),
      legend.key.width = unit(7, "pt"),
      legend.spacing.x = unit(6, "pt")))

save_llm(fig, "output/fig_hidden_trajectory", height_mm = 71,
         panel_ids = c("a", "b", "c"))
