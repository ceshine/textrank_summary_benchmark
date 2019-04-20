library(gt)
library(data.table)

dt <- fread("results.csv")
dt[, c("Rouge-1P", "Rouge-2P", "Rouge-LP") := NULL]
dt[, Model := rep(c("Random", "Head", "TextRank", "BM25+ (eps=.25)", "USE (Base)", "USE (Large)", "USE (Xling)"), 2)]

gt(data = dt, rowname_col = "Model") %>% tab_header(
  title = "CNN / Daily Mail Dataset Benchmark"
) %>% tab_spanner(
    label = "Rouge-1",
    columns = vars("Rouge-1F", "Rouge-1R")
) %>% tab_spanner(
  label = "Rouge-2",
  columns = vars("Rouge-2F", "Rouge-2R")
) %>% tab_spanner(
  label = "Rouge-L",
  columns = vars("Rouge-LF", "Rouge-LR")
) %>% tab_style(
  style = cells_styles(
    text_weight = "bold"
  ),
  locations = cells_data(
    rows = matches("Head")
  )
) %>% tab_style(
  style = cells_styles(
    text_weight = "bold"
  ),
  locations = cells_data(
    columns=vars("Rouge-LR", "Rouge-2R", "Rouge-1R"),
    rows = matches("BM25\\+ \\(eps=.25\\)")
  )
) %>% tab_style(
  style = cells_styles(
    text_weight = "bold"
  ),
  locations = cells_data(
    columns=vars("Rouge-LF", "Rouge-2F", "Rouge-1F"),
    rows = matches("TextRank")
  )
) %>% tab_style(
  style = cells_styles(
    text_color = "#999"
  ),
  locations = cells_data(
    rows = matches("Random")
  )
) %>% cols_label(
  `Rouge-LF` = "Fscore",
  `Rouge-LR` = "Recall",
  `Rouge-2F` = "Fscore",
  `Rouge-2R` = "Recall",
  `Rouge-1F` = "Fscore",
  `Rouge-1R` = "Recall"
) %>% tab_row_group(
  group = "Ratio = .1",
  rows = 1:7
) %>% tab_row_group(
  group = "Ratio = .2",
  rows = 8:14
)