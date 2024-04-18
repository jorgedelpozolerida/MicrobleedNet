# select new cases --------------------------------------------------------
library(tidyverse)
library(readxl)
notannotated_cmb <- read_csv("/home/cerebriu/Downloads/NOTselected_tasks_CRB.csv")
meddare <- read_excel("/home/cerebriu/Downloads/AllPhases_clean.xlsx", na = "NA")

studies_disc_CRB_main <- meddare %>%
  filter(
    grepl("CRB_(Hemo|Infarct|Tumor|include|quality)", discordances) |
      grepl("(infarct|tumor|hemorrhage)_", discordances)
  ) %>%
  select(StudyInstanceUID, discordances)


meddare_tumors <- meddare %>%
  # filter(
  #   !(StudyInstanceUID %in% studies_disc_CRB_main$StudyInstanceUID)
  # ) %>%
  filter(StudyInstanceUID %in% notannotated_cmb$name) %>%
  filter(
    CRB_Tumor == 1
  )

meddare_clean <- meddare %>%
  filter(
    !(StudyInstanceUID %in% studies_disc_CRB_main$StudyInstanceUID)
  ) %>%
  filter(StudyInstanceUID %in% notannotated_cmb$name) %>%
  filter(
    CRB_quality == "sufficient"
  ) %>%
  # filter(
  #   CRB_include == "yes"
  # ) %>%
  filter(
    CRB_Hemorrhage == 1 |
      CRB_Tumor == 1 | CRB_Infarct == 1
  ) %>%
  bind_rows(meddare_tumors) %>%
  arrange(
    desc(CRB_Tumor), (StudyInstanceUID)
  ) %>%
  distinct(StudyInstanceUID, .keep_all = T) %>%
  mutate(priority = 50 - row_number())

meddare_clean_assignees <- meddare_clean %>%
  mutate(
    user_email = "si@cerebriu.com"
  ) %>%
  select(StudyInstanceUID, Dataset, user_email, priority)

meddare_clean_prio <- meddare_clean %>%
  select(StudyInstanceUID, Dataset, priority) %>%
  bind_rows()

write_csv(
  meddare_clean_assignees, "/home/cerebriu/Downloads/pathology_cases_CMB.csv"
)


all_cmbs <- read_excel("/home/cerebriu/Downloads/CMB_datafromMedDARE.xlsx")

newest_meddare_data <- meddare %>%
  filter(StudyInstanceUID %in% all_cmbs$StudyInstanceUID)

writexl::write_xlsx(newest_meddare_data, "/home/cerebriu/Downloads/newest_meddare.xlsx")
