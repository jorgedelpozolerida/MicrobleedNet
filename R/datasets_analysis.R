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

csv_dir <- "/home/cerebriu/data/RESEARCH/MicrobleedNet/data-misc/csv"
cmb_new <- read_csv(file.path(csv_dir, "cmb_new_overview.csv"))
cmb_old <- read_csv(file.path(csv_dir, "cmb_old_overview.csv"))
studies <- read_csv(file.path(csv_dir, "datasets_overview.csv"))


func_getmomeni_patientid <- function(x) {
  xsplit <- str_split(x, "_")
  subjectid <- sapply(xsplit, "[", 1)
  patientid <- paste0(subjectid, "mom")
  return(patientid)
}
func_getmomeni_seriesuid <- function(x, y) {
  xsplit <- str_split(x, "_")
  subjectid <- sapply(xsplit, "[", 1)
  scanid <- sapply(xsplit, "[", 2)
  type <- ifelse(y == "yes", "H", "CMB")
  seriesuid <- paste(subjectid, scanid, sep = "_")
  seriesuid <- paste(seriesuid, type, sep = "-")
  return(seriesuid)
}
func_getmomenisynth_seriesuid <- function(x) {
  xsplit <- str_split(x, "_")
  subjectid <- sapply(xsplit, "[", 1)
  scanid <- sapply(xsplit, "[", 2)
  iteration_id <- sapply(xsplit, function(x) tail(x, 1))
  type <- sapply(sapply(xsplit, function(x) tail(x, 2)), "[", 1)
  seriesuid <- paste(subjectid, scanid, iteration_id, sep = "_")
  seriesuid <- paste(seriesuid, type, sep = "-")

  return(seriesuid)
}
func_valdo_seriesuid <- function(x, y) {
  xsplit <- str_split(x, "-")
  subjectid <- sapply(xsplit, "[", 2)
  type <- ifelse(y == "yes", "H", "CMB")
  seriesuid <- paste(subjectid, sep = "_")
  seriesuid <- paste(seriesuid, type, sep = "-")
  return(seriesuid)
}
func_generic_seriesuid <- function(x, y) {
  type <- ifelse(y == "yes", "H", "CMB")
  seriesuid <- paste(x, sep = "_")
  seriesuid <- paste(seriesuid, type, sep = "-")
  return(seriesuid)
}
func_getvaldo_patientid <- function(x) {
  xsplit <- str_split(x, "-")
  subjectid <- sapply(xsplit, "[", 2)
  patientid <- paste0(subjectid, "val")
  return(patientid)
}
calc_summary <- function(x) {
  # If x is a factor or character, calculate percentages
  if (is.factor(x) || is.character(x)) {
    freq <- table(x)
    percent <- round(100 * freq / sum(freq), 2)

    # Check if only one category exists
    if (length(freq) == 1) {
      return(names(freq))
    } else {
      return(paste(paste0(round(percent), "%"), names(freq), sep = "", collapse = ", "))
    }
  }
  # If x is numeric, calculate min-max
  else if (is.numeric(x)) {
    min_val <- min(x, na.rm = TRUE)
    max_val <- max(x, na.rm = TRUE)
    # Check if min and max are the same
    if (min_val == max_val) {
      return(paste0(min_val))
    } else {
      return(paste0(min_val, "-", max_val))
    }
  }
  # Return NA for other types or if unable to calculate
  else {
    return(NA)
  }
}

## Select studies to be used ---------------------------------------------

# NOTE: idea is that I filter out already here anything not to be used (after manual inspection)

## Clean data ------------------------------------------------------------
studies_clean <- studies %>%
  mutate(Dataset = sub("^p", "", Dataset)) %>%
  mutate(studyUID_old = subject) %>%
  mutate(
    patientUID = case_when(
      str_detect(tolower(Dataset), "momeni") ~ func_getmomeni_patientid(subject),
      str_detect(tolower(Dataset), "valdo") ~ func_getvaldo_patientid(subject),
      TRUE ~ subject
    ),
    series = case_when(
      Dataset == "MOMENI" ~ func_getmomeni_seriesuid(subject, healthy),
      Dataset == "MOMENI_synth" ~ func_getmomenisynth_seriesuid(subject),
      Dataset == "VALDO" ~ func_valdo_seriesuid(subject, healthy),
      TRUE ~ func_generic_seriesuid(subject, healthy)
    )
  ) %>%
  mutate(
    seriesUID = paste(series, Dataset, sep = "-")
  ) %>%
  relocate(seriesUID, n_CMB_new, n_CMB_old) %>%
  mutate(
    Dataset = case_when(
      Dataset == "MOMENI_synth" ~ "sMOMENI",
      Dataset == "CEREBRIU_neg" ~ "CRBneg",
      Dataset == "CEREBRIU" ~ "CRB",
      T ~ Dataset
    )
  ) %>%
  group_by(Dataset) %>%
  arrange(patientUID, series) %>%
  mutate(n_indataset = row_number()) %>%
  ungroup() %>%
  mutate(
    seriesUID = paste(Dataset, n_indataset, series, sep = "-")
  ) %>%
  mutate(
    res_level = sapply(old_voxel_dim, function(x) {
      nums <- as.numeric(unlist(str_extract_all(x, "\\d+\\.\\d+")))
      if (length(nums) >= 2 && all(nums[1:2] > 0.5)) {
        "low"
      } else {
        "high"
      }
    })
  ) %>% 
  # detect CMBs missed in some of the reads (to filter out)
  group_by(Dataset, patientUID) %>% 
  mutate(newCMB=ifelse(length(unique(healthy == "yes"))>1, T, F)) %>% 
  ungroup() %>% 
  # detect CMBs number in some of the reads (to filter out)
  group_by(Dataset, patientUID) %>% 
  mutate(diffCMB=ifelse(length(unique(n_CMB_new))>1, T, F)) %>% 
  ungroup() %>% 
  relocate(seriesUID, patientUID, n_CMB_new, Dataset, subject, seq_type, res_level, healthy) %>% 
  select(-studyUID_old, -series) %>% 
  # Create more binary fields to stritify the training
  group_by(Dataset, patientUID) %>% 
  mutate(
    healthy_all = all(healthy == "yes"),
    nCMB_avg = mean(n_CMB_new)
  ) %>% 
  ungroup() %>% 
  mutate(
    CMB_level = ifelse(nCMB_avg > 3, "high", "low") # based on statistics observed and clinical relevance
  ) %>% 
  # Add scan params
  left_join(tibble(
    Dataset = c("CRB", "CRBneg", "DOU", "MOMENI", "RODEJA", "VALDO", "sMOMENI"),
    field_strength = c("1.5/3", "1.5", "3", "3", "1.5/3", "1.5/3", "3"),
    TE = c(32.5, 27.1, 24.0, 20.0, NA, 25.0, 20.0)
  ), by="Dataset")

problematic_cmb <- studies_clean %>% 
  filter(diffCMB==T)



# Scan parameters building ------------------------------------------------

momeni_scan_params <- data.frame(
  field_strength = "3T",
  scanner_model = "Siemens TRIM TRIO scanner",
  flip_angle = "20",
  TR = 27,
  TE = 20,
  slice_thickness = 1.75,
  rating_scale = "MARS",
  demographics = "Alzheimer’s disease (AD), mild cognitive impairment (MCI) and cognitively normal (CN)",
  location = "Australia"
) %>% mutate(Dataset = "MOMENI")

dou_scan_params <- data.frame(
  field_strength = "3T",
  scanner_model = " Philips Medical System",
  flip_angle = "20",
  TR = 17,
  TE = 24,
  slice_thickness = 2,
  rating_scale = "MARS",
  demographics = "10 cases with stroke and 10 cases of normal aging",
  location = "China?" # clarify with author
) %>% mutate(Dataset = "DOU")

# mix of definite and possible
# The N4 bias field correction technique was applied on all the SWI dataset (JORGE: is applied?)
# where SWI were reconstructed online using the scanner system (software VB17)


valdo_scan_params <- data.frame(
  Study = c("SABRE", "RSS", "ALFA"),
  demographics = c(
    "Tri-ethnic, high cardiovascular risk, 36-92 years old, mean age 72. ",
    "Aging population >45 without dementia",
    "Enriched for APOE4, family risk of Alzheimer’s.  cognitively normal participants aged 45-74"
  ),
  location = c("London, UK", "Rotterdam, Netherlands", "Barcelona, Spain"),
  field_strength = c("3T", "1.5T", "3T"),
  scanner_model = c("Philips", "GE MRI", "GE Discovery"),
  flip_angle = c("18", "13", "15"), # Removing the degree symbol for numerical analysis
  TR = c(1288, 45, 1300),
  TE = c(21, 31, 23),
  slice_thickness = c(3.0, 0.8, 3.0), # Assuming the last dimension in Voxel_Size_mm3 is the slice thickness
  rating_scale = c("BOMBS", "(Vernooij et al., 2008)", "BOMBS"), # Placeholder, as this data is not provided
  stringsAsFactors = FALSE # To avoid automatic conversion to factors
) %>% mutate(Dataset = "VALDO")
# SABRE: initially recruited in 1988 with the purpose of investigating metabolic and cardiovascular
# diseases across ethnicities
# RSS: population-based study that aims to investigate chronic illness in theelderly
# ALFA: details of relatives (generally offspring) of patients with Alzheimer’s Disease making up for a cohort naturally enriched for genetic predisposition to AD


rodeja_scan_params <- data.frame(
  field_strength = "1.5/3T",
  scanner_model = "several",
  flip_angle = "several",
  TR = "several",
  TE = "several",
  slice_thickness = "several",
  rating_scale = "unknown",
  demographics = "unknown", # clarify with author
  location = "Copenhagen region, Denmark"
) %>% mutate(Dataset = "RODEJA")

#### cerebriu_scan_params - negative
crbr_study_metadata <- read_csv("/home/cerebriu/data/DM/MyCerebriu/Pathology_Overview/all_studies_final.csv")

cerebriu_data <- read_csv("/home/cerebriu/data/RESEARCH/MicrobleedNet/data-misc/csv/cerebriu_metadata.csv") %>%
  filter(StudyInstanceUID %in% studies_clean$subject) %>%
  left_join(studies_clean %>% select(subject, Dataset), by = c("StudyInstanceUID" = "subject")) %>%
  # Add study-level
  left_join(
    crbr_study_metadata %>%
      select(-Dataset, -Step),
    by = ("StudyInstanceUID")
  )

cerebriu_scan_params <- cerebriu_data %>%
  group_by(Dataset, Hospital) %>%
  mutate(
    country = sapply(str_split(Dataset, "-"), `[`, 1),
    country = case_when(
      country == "BR" ~ "Brazil",
      country == "IN" ~ "India",
      country == "US" ~ "U.S.A"
    ),
    MagneticFieldStrength = round(as.numeric(MagneticFieldStrength), 2),
    MagneticFieldStrength = case_when(
      MagneticFieldStrength == "15000" ~ 1.5,
      TRUE ~ MagneticFieldStrength
    ),
    Demographics = "Not available",
    field_strength = MagneticFieldStrength,
  ) %>%
  mutate(scanner_model = paste0(Manufacturer, " ", ManufacturerModelName, " ", field_strength)) %>%
  summarise(
    # Demographics = calc_summary(Demographics),
    location = calc_summary(Location),
    scanner_model = calc_summary(scanner_model),
    field_strength = calc_summary(field_strength),
    TR_TE = calc_summary(`TR/TE (ms)`),
    TR = calc_summary(`TR (ms)`),
    TE = calc_summary(`TE (ms)`),
    flip_angle = calc_summary(`Flip Angle`)
  ) %>%
  ungroup() %>%
  mutate(rating_scale = "BOMBS", demographics = "unknown")


# Convert all columns in each dataset to character type
valdo_scan_params[] <- lapply(valdo_scan_params, as.character)
rodeja_scan_params[] <- lapply(rodeja_scan_params, as.character)
cerebriu_scan_params[] <- lapply(cerebriu_scan_params, as.character)
momeni_scan_params[] <- lapply(momeni_scan_params, as.character)
dou_scan_params[] <- lapply(dou_scan_params, as.character)

# Combine all the datasets into one dataframe
all_scan_params <- bind_rows(
  valdo_scan_params,
  rodeja_scan_params,
  cerebriu_scan_params,
  momeni_scan_params,
  dou_scan_params
) %>%
  mutate(Study = paste0(Dataset, ifelse(is.na(Study), "", paste0("-", Study)))) %>%
  select(-Hospital) %>%
  mutate(
    TR_TE = paste0(TR, "/", TE)
  ) %>%
  mutate(
    field_strength = str_replace_all(field_strength, "-", "/"), # Replace - with /
    field_strength = str_replace(field_strength, "[A-Za-z]+$", "") # Remove letters at the end
  )
all_scan_params <- all_scan_params %>%
  # Duplicate rows where Dataset is "MOMENI", changing Dataset to "sMOMENIrs" and "sMOMENIs"
  bind_rows(
    all_scan_params %>%
      filter(Dataset == "MOMENI") %>%
      mutate(Dataset = "sMOMENI"), # Duplicate with modified Dataset
  ) %>%
  mutate(Dataset2 = sapply(str_split(Dataset, "-"), function(x) x[1])) %>%
  relocate(Dataset, Study)

studies_clean %>% 
  group_by(field_strength, res_level, seq_type, healthy_all, CMB_level) %>% 
  summarise(n=n())

write_csv(studies_clean, "/home/cerebriu/data/RESEARCH/MicrobleedNet/data-misc/csv/ALL_studies.csv")





# Summaries ---------------------------------------------------------------

studies_clean_real <- studies_clean %>% filter(Dataset != "sMOMENI")


##########
# Studies, patients
##########

# Summaries by dataset for all studies
summ_studies_dat_seq <- studies_clean %>%
  group_by(Dataset, seq_type) %>%
  summarise(
    n_patients = n_distinct(patientUID),
    n_patients_cmb = n_distinct(patientUID[healthy == "no"]),
    n_patients_h = n_distinct(patientUID[healthy == "yes"]),
    n_series = n_distinct(seriesUID),
    n_series_cmb = n_distinct(seriesUID),
    n_series_h = n_distinct(seriesUID[healthy == "yes"]),
    n_CMB = sum(n_CMB_new, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  ungroup() %>%
  # Adding totals
  bind_rows(
    studies_clean %>%
      summarise(
        Dataset = "Total",
        seq_type = "-",
        n_patients = n_distinct(patientUID),
        n_patients_h = n_distinct(patientUID[healthy == "yes"]),
        n_series = n_distinct(seriesUID),
        n_series_h = sum(healthy == "yes" & !is.na(seriesUID), na.rm = TRUE),
        n_CMB = sum(n_CMB_new, na.rm = TRUE)
      )
  )

summ_studies_dat_res <- studies_clean %>%
  group_by(Dataset, res_level) %>%
  summarise(
    n_patients = n_distinct(patientUID),
    n_patients_h = n_distinct(patientUID[healthy == "yes"]),
    n_series = n_distinct(seriesUID),
    n_series_h = n_distinct(seriesUID[healthy == "yes"]),
    n_CMB = sum(n_CMB_new, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  ungroup() %>%
  # Adding totals
  bind_rows(
    studies_clean %>%
      summarise(
        Dataset = "Total",
        res_level = "-",
        n_patients = n_distinct(patientUID),
        n_patients_h = n_distinct(patientUID[healthy == "yes"]),
        n_series = n_distinct(seriesUID),
        n_series_h = sum(healthy == "yes" & !is.na(seriesUID), na.rm = TRUE),
        n_CMB = sum(n_CMB_new, na.rm = TRUE)
      )
  )

summ_studies_dat_res_seq <- studies_clean %>%
  group_by(Dataset, res_level, seq_type) %>%
  summarise(
    n_patients = n_distinct(patientUID),
    n_patients_h = n_distinct(patientUID[healthy == "yes"]),
    n_series = n_distinct(seriesUID),
    n_series_h = n_distinct(seriesUID[healthy == "yes"]),
    n_CMB = sum(n_CMB_new, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  ungroup() %>%
  # Adding totals
  bind_rows(
    studies_clean %>%
      summarise(
        Dataset = "Total",
        res_level = "-",
        seq_type = "-",
        n_patients = n_distinct(patientUID),
        n_patients_h = n_distinct(patientUID[healthy == "yes"]),
        n_series = n_distinct(seriesUID),
        n_series_h = sum(healthy == "yes" & !is.na(seriesUID), na.rm = TRUE),
        n_CMB = sum(n_CMB_new, na.rm = TRUE)
      )
  )


# summ_studies_dat_res_seq <- studies_clean %>%
#   group_by(Dataset, res_level, seq_type, field_stregth) %>%
#   summarise(
#     n_patients = n_distinct(patientUID),
#     n_patients_h = n_distinct(patientUID[healthy == "yes"]),
#     n_series = n_distinct(seriesUID),
#     n_series_h = n_distinct(seriesUID[healthy == "yes"]),
#     n_CMB = sum(n_CMB_new, na.rm = TRUE),
#     .groups = 'drop'
#   ) %>%
#   ungroup() %>%
#   # Adding totals
#   bind_rows(
#     studies_clean %>%
#       summarise(
#         Dataset = "Total",
#         res_level="-",
#         seq_type="-",
#         field_stregth="-",
#         n_patients = n_distinct(patientUID),
#         n_patients_h = n_distinct(patientUID[healthy == "yes"]),
#         n_series = n_distinct(seriesUID),
#         n_series_h = sum(healthy == "yes" & !is.na(seriesUID), na.rm = TRUE),
#         n_CMB = sum(n_CMB_new, na.rm = TRUE)
#       )
#   )



##########
# n CMB
##########
filtered_data <- studies_clean %>%
  filter(healthy_all!=T) %>% 
  filter(Dataset != "sMOMENI") %>% 
  distinct(Dataset, patientUID, nCMB_avg, CMB_level)

# Create a histogram of 'n_CMB_new', grouped by 'PATIENTiD'
ggplot(filtered_data, aes(x = nCMB_avg)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black") + # You can adjust the binwidth as needed
  theme_minimal() +
  labs(title = "Histogram of CMB by patientUID",
       x = "CMB",
       y = "Frequency") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Improves readability of x-axis labels if needed


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
# summ_nCMB_real <- studies_clean_real %>%
#   distinct(seriesUID, .keep_all = T) %>%
#   select(seriesUID, n_CMB_new) %>%
#   mutate(numeric_var = replace_na(n_CMB_new, 0)) %>%
#   filter(numeric_var != 0) %>%
#   summarise(
#     Count = n(),
#     Mean = mean(numeric_var, na.rm = TRUE),
#     Median = median(numeric_var, na.rm = TRUE),
#     Std_Deviation = sd(numeric_var, na.rm = TRUE),
#     Min = min(numeric_var, na.rm = TRUE),
#     Max = max(numeric_var, na.rm = TRUE),
#     Range = max(numeric_var, na.rm = TRUE) - min(numeric_var, na.rm = TRUE),
#     IQR = IQR(numeric_var, na.rm = TRUE)
#   )
# 
# ##########
# # size and radius CMB
# ##########
# summ_sizeCMB <- cmb_new %>%
#   distinct(subject, Dataset, .keep_all = T) %>%
#   mutate(numeric_var = replace_na(size, 0)) %>%
#   filter(numeric_var != 0) %>%
#   summarise(
#     Count = n(),
#     Mean = mean(numeric_var, na.rm = TRUE),
#     Median = median(numeric_var, na.rm = TRUE),
#     Std_Deviation = sd(numeric_var, na.rm = TRUE),
#     Min = min(numeric_var, na.rm = TRUE),
#     Max = max(numeric_var, na.rm = TRUE),
#     Range = max(numeric_var, na.rm = TRUE) - min(numeric_var, na.rm = TRUE),
#     IQR = IQR(numeric_var, na.rm = TRUE)
#   )
# summ_sizeCMB_real <- cmb_new %>%
#   filter(Dataset != "pMOMENI_synth") %>%
#   distinct(subject, Dataset, .keep_all = T) %>%
#   mutate(numeric_var = replace_na(size, 0)) %>%
#   filter(numeric_var != 0) %>%
#   summarise(
#     Count = n(),
#     Mean = mean(numeric_var, na.rm = TRUE),
#     Median = median(numeric_var, na.rm = TRUE),
#     Std_Deviation = sd(numeric_var, na.rm = TRUE),
#     Min = min(numeric_var, na.rm = TRUE),
#     Max = max(numeric_var, na.rm = TRUE),
#     Range = max(numeric_var, na.rm = TRUE) - min(numeric_var, na.rm = TRUE),
#     IQR = IQR(numeric_var, na.rm = TRUE)
#   )
# 
# summ_radCMB <- cmb_new %>%
#   distinct(subject, Dataset, .keep_all = T) %>%
#   mutate(numeric_var = replace_na(radius, 0)) %>%
#   filter(numeric_var != 0) %>%
#   summarise(
#     Count = n(),
#     Mean = mean(numeric_var, na.rm = TRUE),
#     Median = median(numeric_var, na.rm = TRUE),
#     Std_Deviation = sd(numeric_var, na.rm = TRUE),
#     Min = min(numeric_var, na.rm = TRUE),
#     Max = max(numeric_var, na.rm = TRUE),
#     Range = max(numeric_var, na.rm = TRUE) - min(numeric_var, na.rm = TRUE),
#     IQR = IQR(numeric_var, na.rm = TRUE)
#   )
# summ_radCMB_real <- cmb_new %>%
#   filter(Dataset != "pMOMENI_synth") %>%
#   distinct(subject, Dataset, .keep_all = T) %>%
#   mutate(numeric_var = replace_na(radius, 0)) %>%
#   filter(numeric_var != 0) %>%
#   summarise(
#     Count = n(),
#     Mean = mean(numeric_var, na.rm = TRUE),
#     Median = median(numeric_var, na.rm = TRUE),
#     Std_Deviation = sd(numeric_var, na.rm = TRUE),
#     Min = min(numeric_var, na.rm = TRUE),
#     Max = max(numeric_var, na.rm = TRUE),
#     Range = max(numeric_var, na.rm = TRUE) - min(numeric_var, na.rm = TRUE),
#     IQR = IQR(numeric_var, na.rm = TRUE)
#   )
# 
# 
# ##########
# # resolutions, scan params, scanner
# ##########
# # Function to calculate percentages and format output
# calc_summary <- function(x) {
#   freq <- table(x)
#   percent <- round(100 * freq / sum(freq), 2)
# 
#   # Check if only one category exists
#   if (length(freq) == 1) {
#     return(names(freq))
#   } else {
#     return(paste(paste0(round(percent), "%"), names(freq), sep = ": ", collapse = ", "))
#   }
# }
# 
# summ_reso <- studies_clean %>%
#   summarise(
#     # # Demographics = calc_summary(Demographics),
#     # Location = calc_summary(Location),
#     # `Scanner Type` = calc_summary(`Scanner Type`),
#     # `Scanner Model` = calc_summary(`Scanner Model`),
#     # `Seq. Type` = calc_summary(`Seq. Type`),
#     # `TR/TE (ms)` = calc_summary(`TR/TE (ms)`),
#     # # `TR (ms)` = calc_summary(`TR (ms)`),
#     # # `TE (ms)` = calc_summary(`TE (ms)`),
#     # `Flip Angle` = calc_summary(`Flip Angle`),
#     Resolution = calc_summary(new_shape),
#     Resolution_old = calc_summary(old_shape),
#     `Voxel Size (mm3)` = calc_summary(new_voxel_dim),
#     `Voxel Size (mm3) - OLD` = calc_summary(old_voxel_dim),
#     `# patients` = n()
#   )
# 
# summ_reso_dat <- studies_clean %>%
#   group_by(Dataset) %>%
#   summarise(
#     # # Demographics = calc_summary(Demographics),
#     # Location = calc_summary(Location),
#     # `Scanner Type` = calc_summary(`Scanner Type`),
#     # `Scanner Model` = calc_summary(`Scanner Model`),
#     # `Seq. Type` = calc_summary(`Seq. Type`),
#     # `TR/TE (ms)` = calc_summary(`TR/TE (ms)`),
#     # # `TR (ms)` = calc_summary(`TR (ms)`),
#     # # `TE (ms)` = calc_summary(`TE (ms)`),
#     # `Flip Angle` = calc_summary(`Flip Angle`),
#     Resolution = calc_summary(new_shape),
#     Resolution_old = calc_summary(old_shape),
#     `Voxel Size (mm3)` = calc_summary(new_voxel_dim),
#     `Voxel Size (mm3) - OLD` = calc_summary(old_voxel_dim),
#     `# patients` = n()
#   )
# 
# summ_reso_real <- studies_clean_real %>%
#   summarise(
#     # # Demographics = calc_summary(Demographics),
#     # Location = calc_summary(Location),
#     # `Scanner Type` = calc_summary(`Scanner Type`),
#     # `Scanner Model` = calc_summary(`Scanner Model`),
#     # `Seq. Type` = calc_summary(`Seq. Type`),
#     # `TR/TE (ms)` = calc_summary(`TR/TE (ms)`),
#     # # `TR (ms)` = calc_summary(`TR (ms)`),
#     # # `TE (ms)` = calc_summary(`TE (ms)`),
#     # `Flip Angle` = calc_summary(`Flip Angle`),
#     Resolution = calc_summary(new_shape),
#     Resolution_old = calc_summary(old_shape),
#     `Voxel Size (mm3)` = calc_summary(new_voxel_dim),
#     `Voxel Size (mm3) - OLD` = calc_summary(old_voxel_dim),
#     `# patients` = n()
#   )
# 
# summ_reso_dat_real <- studies_clean_real %>%
#   group_by(Dataset) %>%
#   summarise(
#     # # Demographics = calc_summary(Demographics),
#     # Location = calc_summary(Location),
#     # `Scanner Type` = calc_summary(`Scanner Type`),
#     # `Scanner Model` = calc_summary(`Scanner Model`),
#     # `Seq. Type` = calc_summary(`Seq. Type`),
#     # `TR/TE (ms)` = calc_summary(`TR/TE (ms)`),
#     # # `TR (ms)` = calc_summary(`TR (ms)`),
#     # # `TE (ms)` = calc_summary(`TE (ms)`),
#     # `Flip Angle` = calc_summary(`Flip Angle`),
#     Resolution = calc_summary(new_shape),
#     Resolution_old = calc_summary(old_shape),
#     `Voxel Size (mm3)` = calc_summary(new_voxel_dim),
#     `Voxel Size (mm3) - OLD` = calc_summary(old_voxel_dim),
#     `# patients` = n()
#   )
