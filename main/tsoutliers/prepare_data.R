library(tidyverse)
library(feather)
setwd("/Users/palermopenano/personal/kaggle_energy")
source("kaggle_energy/main/r_code/tsoutliers.R")

# load csv
df <- read_csv("data/train.csv")

# !!! For testing
# df <- read_csv("data/meters.csv")
# df <- df %>% filter(building_id %in% c(880))

save_scores <- function(mytibble)
  # Save the scores as a feather file as
  # they are calculated
{
  building_id <- distinct(mytibble,building_id)$building_id
  meter_id <- distinct(mytibble,meter)$meter
  path <- paste0(
    "/Users/palermopenano/personal/kaggle_energy/tmp/",
    "outlier_score-building",building_id,"_meter",meter_id)
  print(paste0("Processing building-meter: ", building_id, "-", meter_id))
  write_feather(mytibble, path)
}

# Generate a tibble for each group containing the outlier scores
# building id, meter id, and timestamp. Saves results to a feather file
a <-df %>%
  group_by(building_id, meter) %>%
  group_map(~ {
    tibble(
      building_id = .x$building_id,
      meter = .x$meter,
      timestamp = .x$timestamp,
      outliers = tsoutliers(.x$meter_reading, plot=FALSE, lower_prob=0.05, upper_prob=0.95)
    ) %>% save_scores()
}, keep=TRUE)
