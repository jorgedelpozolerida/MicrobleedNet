################################################################################
#' Title:
#' Author: Jorge del Pozo Lerida
#' Date: 2024-02-20
#' Description:
################################################################################

## Setup ----------------------------- --------------------------------------

# Load necessary packages
library(tidyverse)
library(e1071)

set.seed(42)

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
  # Split the input string using underscore as the delimiter
  xsplit <- str_split(x, "_", simplify = TRUE)
  
  # Extract the necessary components based on their positions
  subjectid <- xsplit[, 1]  # First element from each split
  scanid <- xsplit[, 2]     # Second element from each split
  iteration_id <- xsplit[, length(xsplit[1, ])]  # Last element from each split
  type <- xsplit[, length(xsplit[1, ]) - 1]      # Second last element from each split
  
  # Construct seriesuid with the required format
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
      return(paste(paste0(round(percent), "% "), names(freq), sep = "", collapse = ", "))
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

func_get_unique_vals <- function(x) {
  if (is.character(x) && all(grepl("^\\d+", x))) {
    # Assuming numeric-like strings need collapsing into unique, comma-separated lists
    if (n_distinct(x) > 1) {
      paste(unique(x), collapse = ", ")
    } else {
      unique(x)
    }
  } else if (is.numeric(x)) {
    # For numeric data, you might want a simple summary, like mean
    round(mean(x, na.rm = TRUE), 2)
  } else {
    # Default to collapsing into a comma-separated list if multiple unique values
    if (n_distinct(x) > 1) {
      paste(unique(x), collapse = ", ")
    } else {
      unique(x)
    }
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
      Dataset == "sMOMENI" ~ func_getmomenisynth_seriesuid(subject),
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
  mutate(n_CMB_new2 = ifelse(n_CMB_new==0, "", paste0("-", n_CMB_new))) %>% 
  mutate(
    seriesUID = paste0(Dataset,"-", series,  n_CMB_new2)
  ) %>%
    select(-n_CMB_new2) %>% 
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
  # Create more binary fields to stritify the training
  group_by(Dataset, patientUID) %>% 
  mutate(
    healthy_all = all(healthy == "yes"),
    nCMB_avg = mean(n_CMB_new)
  ) %>% 
  ungroup() %>% 
  mutate(
    CMB_level = ifelse(nCMB_avg > 3, "high", "low"), # based on clinical relevance
    CMB_level = ifelse(nCMB_avg == 0, NA,CMB_level )
  ) %>% 
  # Add scan params
  left_join(tibble(
    Dataset = c("CRB", "CRBneg", "DOU", "MOMENI", "RODEJA", "VALDO", "sMOMENI"),
    field_strength = c("1.5/3", "1.5", "3", "3", "1.5/3", "1.5/3", "3"),
    TE = c(32.5, 27.1, 24.0, 20.0, NA, 25.0, 20.0)
  ), by="Dataset") %>% 
  relocate(
    seriesUID, series, n_CMB_new, seq_type,	res_level,	healthy, healthy_all, field_strength,	TE, subject
  )

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
  demographics = "Alzheimer’s disease, mild cognitive impairment and cognitively normal",
  location = "Australia"
) %>% mutate(Dataset = "MOMENI") %>% 
  mutate(  seq_type="SWI")

dou_scan_params <- data.frame(
  field_strength = "3T",
  scanner_model = " Philips Medical System",
  flip_angle = "20",
  TR = 17,
  TE = 24,
  slice_thickness = 2,
  rating_scale = "MARS",
  demographics = "stroke and normal aging",
  location = "Hong Kong" 
) %>% mutate(Dataset = "DOU") %>% 
  mutate(  seq_type="SWI")

# mix of definite and possible
# The N4 bias field correction technique was applied on all the SWI dataset (JORGE: is applied?)
# where SWI were reconstructed online using the scanner system (software VB17)


valdo_scan_params <- data.frame(
  demographics = c(
    "SABRE: Tri-ethnic, high cardiovascular risk, 36-92 years old, mean age 72. ",
    "RSS: Aging population >45 without dementia",
    "ALFA: Enriched for APOE4, family risk of Alzheimer’s.  Cognitively normal participants aged 45-74"
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
) %>%
  mutate(Dataset = "VALDO") %>% 
  mutate(  seq_type="T2S")
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
  location = "Copenhagen region, Denmark",
  seq_type="SWI"
  
) %>% mutate(Dataset = "RODEJA")

#### cerebriu_scan_params - negative

cerebriu_data <- read_csv("/home/cerebriu/data/RESEARCH/MicrobleedNet/data-misc/csv/cerebriu_metadata.csv") %>%
  filter(StudyInstanceUID %in% studies_clean$subject) %>%
  left_join(studies_clean %>% select(subject, Dataset), by = c("StudyInstanceUID" = "subject"))

# metadata 
cerebriu_meddare_extra <- read_csv(
  "/home/cerebriu/data/DM/MyCerebriu/MedDARE/MedDare_documenthandling/datain/marko/MedDARE_Merged_sequence_level.csv"
) %>% 
  filter(StudyInstanceUID %in% studies_clean$subject) %>% 
  filter(
    grepl("SWI|T2S", CRBSeriesDescription)
    ) %>% 
  left_join(
    read_csv(
      "/home/cerebriu/data/DM/MyCerebriu/MedDARE/MedDare_documenthandling/datain/marko/MedDARE_Merged_patient_level.csv"
    ), by=c("StudyInstanceUID", "Dataset")
  ) %>% 
  left_join(
    read_csv("/home/cerebriu/data/DM/MyCerebriu/MedDARE/MedDare_documenthandling/datain/marko/MedDARE_Merged_study_level.csv")
  , by=c("StudyInstanceUID", "Dataset", "PatientID")
  ) %>% 
  rename(Dataset_crb=Dataset) %>% 
  select(
    StudyInstanceUID, SeriesInstanceUID, CRBSeriesDescription, SliceThickness, 
    SpacingBetweenSlices, PatientID, PatientAge, PatientSex, Manufacturer, ManufacturerModelName, MagneticFieldStrength, Dataset_crb
  ) %>% 
  inner_join(cerebriu_data %>% select(StudyInstanceUID, "Seq. Type"), by = c("StudyInstanceUID" = "StudyInstanceUID", 
                                   "CRBSeriesDescription"  ="Seq. Type"))

cerebriu_data <- cerebriu_data %>% 
  left_join(
    cerebriu_meddare_extra, by = "StudyInstanceUID"
  )

cerebriu_scan_params <- cerebriu_data %>%
  mutate(Hospital = Dataset_crb) %>% 
  mutate(PatientAge=case_when(
    PatientAge<18 ~ 18, 
    PatientAge>90 ~ 90,
    T ~ PatientAge
    
    )) %>% 
  # group_by(Dataset, Hospital) %>%
  mutate(
    country = sapply(str_split(Dataset_crb, "-"), `[`, 1),
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
    field_strength = MagneticFieldStrength,
  ) %>%
  group_by(Dataset, country) %>%
  mutate(
    Manufacturer = replace_na(Manufacturer, ""),
    ManufacturerModelName = replace_na(ManufacturerModelName, "")
  ) %>%  mutate(scanner_model = paste0(Manufacturer, " ", ManufacturerModelName)) %>%
  mutate(PatientSex = ifelse(PatientSex == "M", "male", "female")) %>% 
  summarise(
    # Demographics = calc_summary(Demographics),
    location = calc_summary(Location),
    scanner_model = func_get_unique_vals(scanner_model),
    seq_type=func_get_unique_vals(`Seq. Type`),
    field_strength = calc_summary(field_strength),
    slice_thickness = calc_summary(SliceThickness),
    TR_TE = calc_summary(`TR/TE (ms)`),
    TR = calc_summary(`TR (ms)`),
    TE = calc_summary(`TE (ms)`),
    flip_angle = calc_summary(`Flip Angle`),
    Age = calc_summary(PatientAge),
    Sex = calc_summary(PatientSex),
    Age_mean = round(mean(PatientAge, na.rm =T))
  ) %>%
  mutate(
    demographics = paste0(
      "Ages between ", Age, " and mean ", Age_mean, ", with ", Sex
    )
  ) %>% 
  ungroup() %>%
  mutate(rating_scale = "BOMBS") %>% 
  select(-Age, -Sex, -Age_mean, -country)


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
  mutate(
    TR_TE = paste0(TR, "/", TE)
  ) %>%
  mutate(
    field_strength = str_replace_all(field_strength, "-", "/"), # Replace - with /
    field_strength = str_replace(field_strength, "[A-Za-z]+$", "") # Remove letters at the end
  )
all_scan_params <- all_scan_params %>%
  # Duplicate rows where Dataset is "MOMENI", changing Dataset to "sMOMENIrs" and "sMOMENIs"
  mutate(Dataset2 = sapply(str_split(Dataset, "-"), function(x) x[1])) %>%
  select(
    Dataset, location, demographics, scanner_model, seq_type, field_strength,  TR, TE, flip_angle, slice_thickness, rating_scale
    
  )

print(studies_clean %>% 
  group_by(Dataset,field_strength, res_level, seq_type, healthy_all, CMB_level) %>% 
  summarise(n=n()), n=50)

write_csv(studies_clean, "/home/cerebriu/data/RESEARCH/MicrobleedNet/data-misc/csv/ALL_studies.csv")
write_csv(all_scan_params, "/home/cerebriu/data/RESEARCH/MicrobleedNet/data-misc/overviews/all_scan_params.csv")