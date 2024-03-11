################################################################################
#' Title: 
#' Author: Jorge del Pozo Lerida
#' Date: 2024-03-03
#' Description: 
################################################################################

## Setup -------------------------------------------------------------------

# Load necessary packages
library(tidyverse)
library(readxl)


all_phases_merged_raw <- readxl::read_excel(
  "/home/cerebriu/data/DM/MyCerebriu/MedDARE/MedDare_documenthandling/dataout/MedDARE/AllPhases_merged.xlsx",
  na="NA")

all_phases_merged <- all_phases_merged_raw %>% 
  mutate(across(everything(), tolower, .names = "lc_{.col}"))


patterns <- c("microbleeds", "micro-bleeds", "micro bleeds", 
              "microhemorrhages", "micro-hemorrhages", 
              "micro hemorrhages", "microbleedopathy")

# Initialize an empty data frame to store results
results <- data.frame(id = character(),
                      pattern_found = character(), # Ensure this matches your rbind structure
                      column_name = character(),
                      value = character(),
                      stringsAsFactors = FALSE)

# Loop over each column (except 'id') and pattern to find matches
for (col in colnames(all_phases_merged)) {
  if (col != "id") {
    for (pattern in patterns) {
      matched_rows <- which(str_detect(all_phases_merged[[col]], regex(pattern, ignore_case = TRUE)))
      
      # Add found matches to results dataframe
      if (length(matched_rows) > 0) {
        for (row in matched_rows) {
          matched_value <- all_phases_merged[[col]][row]
          # Extract a piece of string around the matched pattern for context
          context_str <- str_extract(matched_value, sprintf(".{0,40}%s.{0,40}", pattern))
          results <- rbind(results, data.frame(id = all_phases_merged$id[row],
                                               pattern_found = pattern,
                                               column_name = col,
                                               value = matched_value, 
                                               context=context_str,
                                               stringsAsFactors = FALSE))
        }
      }
    }
  }
}


results_clean  <- results %>% 
  mutate(StudyInstanceUID = sapply(str_split(id, "_"), `[`, 1)) %>% 
  group_by(StudyInstanceUID) %>% 
  mutate(n=n()) %>% 
  ungroup() %>% 
  relocate(id, StudyInstanceUID)
# %>% 
#   filter(str_detect(column_name, "additional_findings"))

results_clean %>% distinct(StudyInstanceUID) %>% nrow()
results_clean %>% filter(value == "microhemorrhages (<10 mm)") %>% distinct(StudyInstanceUID) %>% nrow()

results_distinct <- results_clean %>% 
  distinct(StudyInstanceUID, .keep_all = T)



# Identify negative -------------------------------------------------------

negative_studies <- all_phases_merged_raw %>% 
  filter(!(StudyInstanceUID %in% results_distinct$StudyInstanceUID)) %>% 
  mutate(
    CRB_quality_suff = ifelse(
      CRB_quality %in% c("sufficient", "insufficient quality (other sequences)"),
      T, F
    )
  ) %>% 
  group_by(StudyInstanceUID) %>% 
  mutate(anybad=any(CRB_quality_suff == FALSE)) %>% 
  ungroup() %>% 
  filter(!anybad)
negative_studies_distinct <- negative_studies %>% 
  distinct(StudyInstanceUID, .keep_all = T)
write_csv(negative_studies, "/home/cerebriu/data/DM/MyCerebriu/CMB/negative_studies_duplicated.csv" )
write_csv(negative_studies_distinct %>% select(StudyInstanceUID),  "/home/cerebriu/data/DM/MyCerebriu/CMB/UIDs_negative_studies_distinct.csv" )
