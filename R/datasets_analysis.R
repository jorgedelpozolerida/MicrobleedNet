################################################################################
#' Title: 
#' Author: Jorge del Pozo Lerida
#' Date: 2024-02-20
#' Description: 
################################################################################

## Setup -------------------------------------------------------------------

# Load necessary packages
library(tidyverse)
library(e1071)
cmb_new <- read_csv("/home/cerebriu/data/RESEARCH/MicrobleedNet/data-misc/csv/cmb_new_overview.csv")
cmb_old <- read_csv("/home/cerebriu/data/RESEARCH/MicrobleedNet/data-misc/csv/cmb_old_overview.csv")
studies <- read_csv("/home/cerebriu/data/RESEARCH/MicrobleedNet/data-misc/csv/datasets_overview.csv")


func_getmomeni_patientid <- function(x){
  xsplit <- str_split(x, "_")
  subjectid <- sapply(xsplit, '[', 1)
  return(subjectid)
}
func_getmomeni_seriesuid <- function(x){
  xsplit <- str_split(x, "_")
  subjectid <- sapply(xsplit, '[', 1)
  scanid <- sapply(xsplit, '[', 2)
  seriesuid <- paste(subjectid, scanid, sep="_")
  return(seriesuid)
}
func_getvaldo_patientid <- function(x){
  xsplit <- str_split(x, "-")
  subjectid <- sapply(xsplit, '[', 2)
  return(subjectid)
}

## Select studies to be used ---------------------------------------------

# NOTE: idea is that I filter out already here anything not to be used (after manual inspection)



## Clean data ------------------------------------------------------------
studies_clean <- studies %>% 
  mutate(dat=Dataset) %>% 
  group_by(Dataset) %>% 
  mutate(
    patient = case_when(
      str_detect(tolower(dat), "momeni") ~ func_getmomeni_patientid(subject),
      str_detect(tolower(dat), "valdo") ~ func_getvaldo_patientid(subject),
      TRUE ~ subject
    ),
    series = case_when(
      str_detect(tolower(dat), "momeni")  ~ func_getmomeni_seriesuid(subject),
      TRUE ~ subject
    )
  ) %>%
  ungroup() %>% 
  group_by(Dataset, patient) %>% 
  arrange(series) %>% 
  mutate(seriesUID = row_number()) %>% 
  ungroup() %>% 
  group_by(Dataset) %>%
  arrange(patient) %>% 
  mutate(patientUID = row_number()) %>% 
  ungroup() %>% 
  mutate(patientUID = case_when(
    str_detect(tolower(dat), "momeni") ~ paste0(patientUID, "-", "momeni"),
    TRUE ~ paste0(patientUID, "-", dat))
            ) %>% 
  mutate(n = row_number()) %>% 
  mutate(seriesUID = paste0(n,"-", dat, "-", seriesUID) ) %>%
  arrange(Dataset) %>% 
  mutate(id = row_number()) %>% 
  relocate(id, seriesUID, patientUID) %>% 
  select(-dat, -patient, -series, -n) %>% 
  group_by(patientUID) %>% 
  mutate(patient_scan_num = row_number()) %>% 
  ungroup()
  
studies_clean_real <- studies_clean %>% filter(Dataset!="pMOMENI_synth")





# Summaries ---------------------------------------------------------------

##########
# Studies, patients
##########
summ_studies <- data.frame(
  n_patients = studies_clean %>% distinct(patientUID) %>% nrow(),
  n_patients_h = studies_clean %>% filter(healthy == "yes") %>% distinct(patientUID) %>% nrow(),
  n_series = studies_clean %>% distinct(seriesUID) %>% nrow(),
  n_series_h = studies_clean %>% filter(healthy == "yes") %>% distinct(seriesUID) %>% nrow(),
  n_CMB = studies_clean  %>% select(n_CMB_new) %>% unlist() %>% as.numeric() %>% sum()
)
# Summaries by dataset for all studies
summ_studies_dat <- studies_clean %>%
  group_by(Dataset) %>%
  summarise(
    n_patients = n_distinct(patientUID),
    n_patients_h = sum(healthy == "yes", na.rm = TRUE),
    n_series = n_distinct(seriesUID),
    n_series_h = sum(healthy == "yes" & !is.na(seriesUID), na.rm = TRUE),
    n_CMB = sum(n_CMB_new, na.rm = TRUE),
    .groups = 'drop'
  )

summ_studies_nosynth <- data.frame(
  n_patients = studies_clean_real %>% distinct(patientUID) %>% nrow(),
  n_patients_h = studies_clean_real %>% filter(healthy == "yes") %>% distinct(patientUID) %>% nrow(),
  n_series = studies_clean_real %>% distinct(seriesUID) %>% nrow(),
  n_series_h = studies_clean_real %>% filter(healthy == "yes") %>% distinct(seriesUID) %>% nrow(),
  n_CMB = studies_clean_real  %>% select(n_CMB_new) %>% unlist() %>% as.numeric() %>% sum()
)
# Summaries by dataset for no synthetic (real) studies
summ_studies_nosynth_dat <- studies_clean_real %>%
  group_by(Dataset) %>%
  summarise(
    n_patients = n_distinct(patientUID),
    n_patients_h = sum(healthy == "yes", na.rm = TRUE),
    n_series = n_distinct(seriesUID),
    n_series_h = sum(healthy == "yes" & !is.na(seriesUID), na.rm = TRUE),
    n_CMB = sum(n_CMB_new, na.rm = TRUE),
    .groups = 'drop'
  )



##########
# n CMB
##########
summ_nCMB <- studies_clean %>% 
  distinct(seriesUID, .keep_all = T) %>% 
  select(seriesUID, n_CMB_new) %>% 
  mutate(numeric_var = replace_na(n_CMB_new, 0)) %>%
  filter(numeric_var != 0) %>% 
  summarise(
    Count = n(),
    Mean = mean(numeric_var, na.rm = TRUE),
    Median = median(numeric_var, na.rm = TRUE),
    Std_Deviation = sd(numeric_var, na.rm = TRUE),
    Min = min(numeric_var, na.rm = TRUE),
    Max = max(numeric_var, na.rm = TRUE),
    Range = max(numeric_var, na.rm = TRUE) - min(numeric_var, na.rm = TRUE),
    IQR = IQR(numeric_var, na.rm = TRUE)
  )
summ_nCMB_real <- studies_clean_real %>% 
  distinct(seriesUID, .keep_all = T) %>% 
  select(seriesUID, n_CMB_new) %>% 
  mutate(numeric_var = replace_na(n_CMB_new, 0)) %>%
  filter(numeric_var != 0) %>% 
  summarise(
    Count = n(),
    Mean = mean(numeric_var, na.rm = TRUE),
    Median = median(numeric_var, na.rm = TRUE),
    Std_Deviation = sd(numeric_var, na.rm = TRUE),
    Min = min(numeric_var, na.rm = TRUE),
    Max = max(numeric_var, na.rm = TRUE),
    Range = max(numeric_var, na.rm = TRUE) - min(numeric_var, na.rm = TRUE),
    IQR = IQR(numeric_var, na.rm = TRUE)
  )

##########
# size and radius CMB
##########
summ_sizeCMB <- cmb_new %>% 
  distinct(subject, Dataset, .keep_all = T) %>% 
  mutate(numeric_var = replace_na(size, 0)) %>%
  filter(numeric_var != 0) %>% 
  summarise(
    Count = n(),
    Mean = mean(numeric_var, na.rm = TRUE),
    Median = median(numeric_var, na.rm = TRUE),
    Std_Deviation = sd(numeric_var, na.rm = TRUE),
    Min = min(numeric_var, na.rm = TRUE),
    Max = max(numeric_var, na.rm = TRUE),
    Range = max(numeric_var, na.rm = TRUE) - min(numeric_var, na.rm = TRUE),
    IQR = IQR(numeric_var, na.rm = TRUE)
  )
summ_sizeCMB_real <- cmb_new %>% 
  filter(Dataset!="pMOMENI_synth") %>% 
  distinct(subject, Dataset, .keep_all = T) %>% 
  mutate(numeric_var = replace_na(size, 0)) %>%
  filter(numeric_var != 0) %>% 
  summarise(
    Count = n(),
    Mean = mean(numeric_var, na.rm = TRUE),
    Median = median(numeric_var, na.rm = TRUE),
    Std_Deviation = sd(numeric_var, na.rm = TRUE),
    Min = min(numeric_var, na.rm = TRUE),
    Max = max(numeric_var, na.rm = TRUE),
    Range = max(numeric_var, na.rm = TRUE) - min(numeric_var, na.rm = TRUE),
    IQR = IQR(numeric_var, na.rm = TRUE)
  )

summ_radCMB <- cmb_new %>% 
  distinct(subject, Dataset, .keep_all = T) %>% 
  mutate(numeric_var = replace_na(radius, 0)) %>%
  filter(numeric_var != 0) %>% 
  summarise(
    Count = n(),
    Mean = mean(numeric_var, na.rm = TRUE),
    Median = median(numeric_var, na.rm = TRUE),
    Std_Deviation = sd(numeric_var, na.rm = TRUE),
    Min = min(numeric_var, na.rm = TRUE),
    Max = max(numeric_var, na.rm = TRUE),
    Range = max(numeric_var, na.rm = TRUE) - min(numeric_var, na.rm = TRUE),
    IQR = IQR(numeric_var, na.rm = TRUE)
  )
summ_radCMB_real <- cmb_new %>% 
  filter(Dataset!="pMOMENI_synth") %>% 
  distinct(subject, Dataset, .keep_all = T) %>% 
  mutate(numeric_var = replace_na(radius, 0)) %>%
  filter(numeric_var != 0) %>% 
  summarise(
    Count = n(),
    Mean = mean(numeric_var, na.rm = TRUE),
    Median = median(numeric_var, na.rm = TRUE),
    Std_Deviation = sd(numeric_var, na.rm = TRUE),
    Min = min(numeric_var, na.rm = TRUE),
    Max = max(numeric_var, na.rm = TRUE),
    Range = max(numeric_var, na.rm = TRUE) - min(numeric_var, na.rm = TRUE),
    IQR = IQR(numeric_var, na.rm = TRUE)
  )


##########
# resolutions, scan params, scanner
##########
# Function to calculate percentages and format output
calc_percent <- function(x) {
  freq <- table(x)
  percent <- round(100 * freq / sum(freq), 2)
  
  # Check if only one category exists
  if (length(freq) == 1) {
    return(names(freq))
  } else {
    return(paste(paste0(round(percent), "%"), names(freq), sep=": ", collapse=", "))
  }
}

summ_reso <- studies_clean %>%
  summarise(
    # # Demographics = calc_percent(Demographics),
    # Location = calc_percent(Location),
    # `Scanner Type` = calc_percent(`Scanner Type`),
    # `Scanner Model` = calc_percent(`Scanner Model`),
    # `Seq. Type` = calc_percent(`Seq. Type`),
    # `TR/TE (ms)` = calc_percent(`TR/TE (ms)`),
    # # `TR (ms)` = calc_percent(`TR (ms)`),
    # # `TE (ms)` = calc_percent(`TE (ms)`),
    # `Flip Angle` = calc_percent(`Flip Angle`),
    Resolution = calc_percent(new_shape),
    Resolution_old = calc_percent(old_shape),
    `Voxel Size (mm3)` = calc_percent(new_voxel_dim),
    `Voxel Size (mm3) - OLD` = calc_percent(old_voxel_dim),
    
    `# patients` = n()
  )

summ_reso_dat <- studies_clean %>%
  group_by(Dataset) %>%
  summarise(
    # # Demographics = calc_percent(Demographics),
    # Location = calc_percent(Location),
    # `Scanner Type` = calc_percent(`Scanner Type`),
    # `Scanner Model` = calc_percent(`Scanner Model`),
    # `Seq. Type` = calc_percent(`Seq. Type`),
    # `TR/TE (ms)` = calc_percent(`TR/TE (ms)`),
    # # `TR (ms)` = calc_percent(`TR (ms)`),
    # # `TE (ms)` = calc_percent(`TE (ms)`),
    # `Flip Angle` = calc_percent(`Flip Angle`),
    Resolution = calc_percent(new_shape),
    Resolution_old = calc_percent(old_shape),
    `Voxel Size (mm3)` = calc_percent(new_voxel_dim),
    `Voxel Size (mm3) - OLD` = calc_percent(old_voxel_dim),
    
    `# patients` = n()
  )

summ_reso_real <- studies_clean_real %>%
  summarise(
    # # Demographics = calc_percent(Demographics),
    # Location = calc_percent(Location),
    # `Scanner Type` = calc_percent(`Scanner Type`),
    # `Scanner Model` = calc_percent(`Scanner Model`),
    # `Seq. Type` = calc_percent(`Seq. Type`),
    # `TR/TE (ms)` = calc_percent(`TR/TE (ms)`),
    # # `TR (ms)` = calc_percent(`TR (ms)`),
    # # `TE (ms)` = calc_percent(`TE (ms)`),
    # `Flip Angle` = calc_percent(`Flip Angle`),
    Resolution = calc_percent(new_shape),
    Resolution_old = calc_percent(old_shape),
    `Voxel Size (mm3)` = calc_percent(new_voxel_dim),
    `Voxel Size (mm3) - OLD` = calc_percent(old_voxel_dim),
    
    `# patients` = n()
  )

summ_reso_dat_real <- studies_clean_real %>%
  group_by(Dataset) %>%
  summarise(
    # # Demographics = calc_percent(Demographics),
    # Location = calc_percent(Location),
    # `Scanner Type` = calc_percent(`Scanner Type`),
    # `Scanner Model` = calc_percent(`Scanner Model`),
    # `Seq. Type` = calc_percent(`Seq. Type`),
    # `TR/TE (ms)` = calc_percent(`TR/TE (ms)`),
    # # `TR (ms)` = calc_percent(`TR (ms)`),
    # # `TE (ms)` = calc_percent(`TE (ms)`),
    # `Flip Angle` = calc_percent(`Flip Angle`),
    Resolution = calc_percent(new_shape),
    Resolution_old = calc_percent(old_shape),
    `Voxel Size (mm3)` = calc_percent(new_voxel_dim),
    `Voxel Size (mm3) - OLD` = calc_percent(old_voxel_dim),
    
    `# patients` = n()
  )
