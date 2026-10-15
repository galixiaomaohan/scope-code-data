# -*- coding: utf-8 -*-
# pub_style_llm.R -- shared style for the llm_analysis figure set.
# Inherits the manuscript R theme (Times New Roman, hairline axes, frameless
# legends) and adds two signal hues used only inside this set (amber = failed /
# attention, teal = mechanism), plus a shared data directory and export naming.

# Per-figure CSV/JSON exports live under data/processed/figures/ in this
# package (run scripts with working directory figures_src/).
here <- function(...) file.path("..", "data", "processed", "figures", ...)

PAL <- c(
  blue_main = "#0F4D92",
  blue_sec  = "#3775BA",
  blue_l    = "#B9D2EA",
  amber     = "#E8A020",
  amber_l   = "#F5D9A0",
  teal      = "#2C8C8C",
  green_s   = "#3E8E55",
  red_s     = "#B64342",
  grey_l    = "#CFCECE",
  grey_m    = "#8A8A8A",
  grey_d    = "#474747",
  ink       = "#262626"
)

# Seven vocabulary segments (head-geometry figure); low-saturation family with
# grey reserved for the legacy base so the "legacy vs added" boundary pops.
# Keys are the display labels (with member counts) used in the legend.
SEG7 <- c(
  "Base (legacy 161)"      = "#8A8A8A",
  "Deep-chain ext. (17)"   = "#E8A020",
  "Constr. curriculum (7)" = "#3E8E55",
  "Deep-chain family (9)"  = "#B64342",
  "Subst. / eval. (4)"     = "#C77DBB",
  "Routing (9)"            = "#3775BA",
  "Expansion (13)"         = "#2C8C8C"
)

FAM3 <- c(
  "Nested induction"       = "#3775BA",
  "Lemma tower"            = "#E8A020",
  "Sum-of-squares cascade" = "#3E8E55"
)

theme_pub <- function(base_size = 7.5) {
  theme_classic(base_size = base_size, base_family = "Times New Roman") +
    theme(
      axis.line         = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks        = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks.length = unit(2, "pt"),
      axis.title        = element_text(size = base_size, colour = "black"),
      axis.text         = element_text(size = base_size - 0.5, colour = "black"),
      legend.title      = element_blank(),
      legend.text       = element_text(size = base_size - 0.5, colour = "black"),
      legend.key        = element_blank(),
      legend.key.width  = unit(9, "pt"),
      legend.background = element_blank(),
      plot.tag          = element_text(size = 9, face = "bold",
                                       family = "Times New Roman"),
      plot.tag.position = "topleft",
      plot.title        = element_text(size = base_size + 0.5, face = "bold",
                                       hjust = 0, colour = "black"),
      strip.background  = element_blank(),
      strip.text        = element_text(size = base_size, face = "bold",
                                       colour = "black")
    )
}

# Every figure in this set is exported at the SAME physical width (double
# column, 183 mm) so the set reads as one family in the manuscript.
WIDTH_MM  <- 183
save_llm <- function(plot, filename, height_mm, dpi = 600, ...) {
  save_pub_r(plot, filename, width_mm = WIDTH_MM, height_mm = height_mm,
             dpi = dpi, ...)
}
