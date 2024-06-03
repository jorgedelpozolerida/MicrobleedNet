
library(tidyverse)
meddare <- read_excel("/home/cerebriu/Downloads/AllPhases_clean.xlsx", na = "NA")
all_Studies <- read_csv("/home/cerebriu/data/RESEARCH/MicrobleedNet/data-misc/csv/ALL_studies.csv") %>% 
  filter(
    Dataset == "CRB"
  ) %>% 
  mutate(
    studyuid = str_split(seriesUID, "-", simplify = T)[, 2]
  ) %>% relocate(studyuid)



studies <- all_Studies %>% pull(studyuid)


meddare_filt <- meddare %>% 
  filter(
    StudyInstanceUID %in% studies
  ) %>% 
  distinct(StudyInstanceUID, .keep_all = T) 


summ_path <- meddare_filt %>% 
  summarise(
    infarct = sum(CRB_Infarct),
    tumor = sum(CRB_Tumor),
    hemo = sum(CRB_Hemorrhage),
    )
  )