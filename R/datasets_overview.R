################################################################################
#' Title: 
#' Author: Jorge del Pozo Lerida
#' Date: 2024-04-27
#' Description: 
################################################################################

## Setup -------------------------------------------------------------------
library(tidyverse)
library(oro.nifti)



all_studies <- read_csv("../data-misc/csv/ALL_studies.csv")


# Summaries ---------------------------------------------------------------


##########
# Studies, patients
##########

# resolution, # numb CMB per patient, # size of CMB


# Summaries by dataset for all studies
summ_studies_dat_seq <- all_studies %>%
  group_by(Dataset) %>%
  summarise(
    n_series = n_distinct(seriesUID),
    n_series_cmb = n_distinct(seriesUID[healthy == "no"]),
    n_patients = n_distinct(patientUID),
    n_patients_cmb = n_distinct(patientUID[healthy == "no"]),
    n_CMB = sum(n_CMB_new, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  ungroup() %>%
  # Adding totals
  bind_rows(
    all_studies %>%
      summarise(
        Dataset = "Total",
        n_series = n_distinct(seriesUID),
        n_series_cmb = sum(healthy == "no" & !is.na(seriesUID), na.rm = TRUE),
        n_patients = n_distinct(patientUID),
        n_patients_cmb = n_distinct(patientUID[healthy == "no"]),

        n_CMB = sum(n_CMB_new, na.rm = TRUE)
      )
  )
# 
# summ_studies_dat_res <- all_studies %>%
#   group_by(Dataset, res_level) %>%
#   summarise(
#     n_patients = n_distinct(patientUID),
#     n_patients_h = n_distinct(patientUID[healthy == "yes"]),
#     n_series = n_distinct(seriesUID),
#     n_series_h = n_distinct(seriesUID[healthy == "yes"]),
#     n_CMB = sum(n_CMB_new, na.rm = TRUE),
#     .groups = "drop"
#   ) %>%
#   ungroup() %>%
#   # Adding totals
#   bind_rows(
#     all_studies %>%
#       summarise(
#         Dataset = "Total",
#         res_level = "-",
#         n_patients = n_distinct(patientUID),
#         n_patients_h = n_distinct(patientUID[healthy == "yes"]),
#         n_series = n_distinct(seriesUID),
#         n_series_h = sum(healthy == "yes" & !is.na(seriesUID), na.rm = TRUE),
#         n_CMB = sum(n_CMB_new, na.rm = TRUE)
#       )
#   )
# 
# summ_studies_dat_res_seq <- all_studies %>%
#   group_by(Dataset, res_level, seq_type) %>%
#   summarise(
#     n_patients = n_distinct(patientUID),
#     n_patients_h = n_distinct(patientUID[healthy == "yes"]),
#     n_series = n_distinct(seriesUID),
#     n_series_h = n_distinct(seriesUID[healthy == "yes"]),
#     n_CMB = sum(n_CMB_new, na.rm = TRUE),
#     .groups = "drop"
#   ) %>%
#   ungroup() %>%
#   # Adding totals
#   bind_rows(
#     all_studies %>%
#       summarise(
#         Dataset = "Total",
#         res_level = "-",
#         seq_type = "-",
#         n_patients = n_distinct(patientUID),
#         n_patients_h = n_distinct(patientUID[healthy == "yes"]),
#         n_series = n_distinct(seriesUID),
#         n_series_h = sum(healthy == "yes" & !is.na(seriesUID), na.rm = TRUE),
#         n_CMB = sum(n_CMB_new, na.rm = TRUE)
#       )
#   )


# summ_studies_dat_res_seq <- all_studies %>%
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
#     all_studies %>%
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
filtered_data <- all_studies %>%
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


summ_nCMB <- all_studies %>%
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
# summ_nCMB_real <- all_studies_real %>%
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
# summ_reso <- all_studies %>%
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
# summ_reso_dat <- all_studies %>%
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
# summ_reso_real <- all_studies_real %>%
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
# summ_reso_dat_real <- all_studies_real %>%
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