# Jorge's Weekly Meeting Notes

* [05 February 2024](#date-05-february-2024)
* [19 February 2024](#date-19-february-2024)
* [26 February 2024](#date-26-february-2024)
* [04 March 2024](#date-04-march-2024)
* [11 March 2024](#date-11-march-2024)
* [18 March 2024](#date-18-march-2024)
* [25 March 2024](#date-25-march-2024)


<!-- *
* [12 February 2024](#date-12-february-2024)
* [26 February 2024](#date-26-february-2024)
* [04 March 2024](#date-04-march-2024)
* [11 March 2024](#date-11-march-2024)
* [01 April 2024](#date-01-april-2024)
* [08 April 2024](#date-08-april-2024)
* [15 April 2024](#date-15-april-2024)
* [22 April 2024](#date-22-april-2024)
* [29 April 2024](#date-29-april-2024)
* [06 May 2024](#date-06-may-2024)
* [13 May 2024](#date-13-may-2024)
* [20 May 2024](#date-20-may-2024)
* [27 May 2024](#date-27-may-2024)
* [03 June 2024](#date-03-june-2024)
* [10 June 2024](#date-10-june-2024)
* [17 June 2024](#date-17-june-2024)
* [24 June 2024](#date-24-june-2024)
* [01 July 2024](#date-01-july-2024)

* [Template](#date-template) -->

<br><br><br><br><br>



### Date: Template

#### Who did you help this week?

Replace this text with a one/two sentence description of who you helped this week and how.


#### What helped you this week?

Replace this text with a one/two sentence description of what helped you this week and how.

#### What did you achieve/do?

* Replace this text with a bullet point list of what you achieved this week.
* It's ok if your list is only one bullet point long!

#### What did you struggle with?

* Replace this text with a bullet point list of where you struggled this week.
* It's ok if your list is only one bullet point long!

#### What would you like to work on next ?

* Replace this text with a bullet point list of what you would like to work on next week.
* It's ok if your list is only one bullet point long!
* Try to estimate how long each task will take.

#### Where do you need help from Veronika?

* Replace this text with a bullet point list of what you need help from Veronica on.
* It's ok if your list is only one bullet point long!
* Try to estimate how long each task will take.

#### Others


This space is yours to add to as needed.

<br><br><br><br><br>
<br><br><br><br><br>


### Date: 29 January 2024

#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>




### Date: 05 February 2024

#### What did you achieve/do?

* Code for pre-processing CEREBRIU annotations finished -> possible improvements in future
* Code for post-processing with SynthSeg created
* Code for evaluating locally created
* Restructured repo with utility functions scripts

* Extra evalaution on CRBR data + Submitted Abstract to KDD

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* KDD paper writing to be sent early in the week
* Code for processing all datasets

#### Where do you need help from Veronika?
* KDD draft feedback

#### Others

* n/a

<br><br><br><br><br>




### Date: 12 February 2024

#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>




### Date: 19 February 2024

(I cannot attend to this meeting)

#### What did you achieve/do?

For KDD (also useful for thesis):
* Code for post-processing with SynthSeg created
* Code for pre-processing CEREBRIU annotations finished
* Code for evaluating locally created
* Extra evaluation on CRBR data
* KDD report

For thesis only:
* Code for pre-processing DOU dataset finished
* Code for pre-processing MOMENI dataset finished
* Restructured repo with python modules as package


#### What did you struggle with?
* Hard to apply Region Growing when contrast is bad, like in MOMENI dataset
* Specific paraemters of region growing to every dataset

#### What would you like to work on next ?
* Polish dataset preprocessing, manually check visually all results
* Code for pre-processing MOMENI-synthetic dataset
* Code for processing KU dataset (i call it "RODEJA")
* Think of how to combine all datasets, specially when
it comes to negative samples and possible use of synthestic dataset from MOMENI

#### Where do you need help from Veronika?
* What is your opinion on the use of synthetic microbleeds as created in this paper?:
[Momeni paper](../docs/papers/Synthetic%20microbleeds%20generation%20for%20classifier%20training%20without.pdf) --> basically there are 3,700 scans from 118 patients with synthetic CMBs modelled with a random Gaussian shape and added to healthy brain locations


#### Others

* I name datasets by the first author of their papers

<br><br><br><br><br>




### Date: 26 February 2024

#### What did you achieve/do?
* Almost finished preprocessigng for: VALDO, DOU, MOMENI, CEREBRIU

#### What did you struggle with?
* Region Growing out of control many times. Had to do a lot of manual inspection...

#### What would you like to work on next ?
* Finish once and for all preprocessing.
* Clean data from RODEJA

#### Where do you need help from Veronika?
* What is your opinion on the use of synthetic microbleeds as created in this paper?:
[Momeni paper](../docs/papers/Synthetic%20microbleeds%20generation%20for%20classifier%20training%20without.pdf) --> basically there are 3,700 scans from 118 patients with synthetic CMBs modelled with a random Gaussian shape and added to healthy brain locations
* In the end Region Growing was not trustworthy, but gave nice estimate of radius. Using this to build a sphere around center. What do you think?
* I am thinking of using a lot of negative data internal to allow model to learn when no microbleeds
* Confirmation form SAP abou t presentation dates?

#### Others

* Rough summary of data:
# Dataset Overview

The following table provides an overview of the datasets, including the number of scans, patients, cerebral microbleeds (CMBs), scanner types, and specifications.

| DATASET        | n_scans_neg | n_scans_CMB | n_patients_neg | n_patients_CMB | n_CMB          | scanner                        | specs                                                                                                         |
|----------------|-------------|-------------|----------------|----------------|----------------|--------------------------------|---------------------------------------------------------------------------------------------------------------|
| VALDO          | 23          | 49          | 23             | 49             | -              | 1GE 1.5T, 2Phillips 3T         | 1low_res, 2high_res                                                                                            |
| DOU            | -           | 20          | -              | 20             | ~80            | 3.0T Philips Medical System    | TR=17ms, TE=24ms, resolution=(512×512×150), voxel=(0.45×0.45x1), slice_thickness=2mm, slice_spacing=1mm, FOV=230×230 mm2 |
| MOMENI         | 313         | 57          | 100            | 30             | 148            | 3T                             | resolution=(0.93×0.93×1.75), TE=20                                                                            |
| MOMENI_rsCMB   | -           | 570         | -              | 30 (same)      | 148 + 570*10 = 5848 | 3T                             | resolution=(0.93×0.93×1.75), TE=20                                                                            |
| MOMENI_sCMB    | -           | 3130        | -              | 100 (same)     | 0 + 3130*10 = 31300 | 3T                             | resolution=(0.93×0.93×1.75), TE=20                                                                            |
| CEREBRIU       | -           | 7/70        | -              | 7/70           | -              | diverse (table below)          | diverse (table below)                                                                                          |
| RODEJA         | 22 + 40?    | 40          | 22 + 40?       | 40             | -              | 1.5T and 3T                    | various resolutions from (0.2×0.2×1) to (1 × 1 × 6)                                                           |
| TOTAL          | 350         | ~200 (+3600)| ~150           | ~150           | -              | -                              | -                                                                                                             |



<br><br><br><br><br>




### Date: 04 March 2024

#### What did you achieve/do?
* Selected 750 negative cases form internal data (not having CMBs supposedly)
* Practically finished preprocessing for all datasets, debug some code error missing


#### What did you struggle with?
* Preprocessing takes incredibly lot of time, also since there are so many studies 
and is very computationally demaning I need to do on server and parallellize everything, 
which makes debugging very complex

#### What would you like to work on next ?
* Add Skull stripping to preprocessing pipeline (thinking of using SynthStrip model)
* Run preprocessing again on everything with skull stripping
* Implement Apollo (CEREBRIU's architecture) in ClearML
* Pretrain on negative + synthetic + real with and without Apollo's weights initialized

#### Where do you need help from Veronika?
* What's your opinion on skull stripping? i wasn't sure whther to leave it as it is or keep brain only

#### Others

* Examples of how microbleeds look after all preprocessing pipeline:
    - DOU dataset: [data-misc/img/03-CMB-2.png](../data-misc/img/03-CMB-2.png)
    - RODEJA dataset: [data-misc/img/00006-CMB-0.png](../data-misc/img/00006-CMB-0.png)
    - MOMENI dataset: [data-misc/img/122_T1_MRI_SWI_BFC_50mm_HM-CMB-1.png](../data-misc/img/122_T1_MRI_SWI_BFC_50mm_HM-CMB-1.png)
    - MOMENI-synth: [data-misc/img/2_T2_MRI_SWI_BFC_50mm_HM_sCMB_V2-CMB-3.png](../data-misc/img/2_T2_MRI_SWI_BFC_50mm_HM_sCMB_V2-CMB-3.png)
    - VALDO [data-misc/img/sub-101-CMB-12.png](../data-misc/img/sub-101-CMB-12.png)


<br><br><br><br><br>




### Date: 11 March 2024

#### What did you achieve/do?
* All datasets preprocessed!! Took a full month but done finally :'). 

#### What did you struggle with?
* During dataset preprocessing many unexpected behaviour could be observed that affected the final count of microbleeds: resampling effect (splitting CMBs-upsampling or joining two together-downsampling), morphological operations messing with masks, region growing having unexpected results, annotations being incorrectly placed...etc But managed to keep original count intact at least for all real CMBs

#### What would you like to work on next ?
* Decide on how to use data for training in best way
* Start pre-training with negative + synthetic + real
* While it trains, start writing Data Preprocessing section, as well as intro and background

#### Where do you need help from Veronika?
* Ideas on how to split data for training-test?

#### Others

* _Processing steps_: mask cleaning (RG, CC decomposition, possibly sphere creation..etc) + reorientation +  skull stripping + cropping  + resampling + CMB count checking
* CSV indicating studies-CMBs where manually set radius size is used (dou and momeni) or where spheres are created instead of using original mask (rodeja): [manual_fixes.csv](../data-misc/processing/manual_fixes.csv)

* Table summarizing all data available:

| Dataset | res_level | seq_type | n_patients | n_patients_h | n_series | n_series_h | n_CMB |
|----------|-----------|----------|------------|--------------|----------|------------|-------|
| CRB      | high      | T2S      |          3 |            0 |        3 |          0 |    37 |
| CRB      | low       | SWI      |          4 |            0 |        4 |          0 |    47 |
| CRB      | low       | T2S      |          1 |            0 |        1 |          0 |    12 |
| CRBneg   | high      | SWI      |          1 |            1 |        1 |          1 |     0 |
| CRBneg   | high      | T2S      |        442 |          442 |      442 |        442 |     0 |
| CRBneg   | low       | SWI      |        200 |          200 |      200 |        200 |     0 |
| CRBneg   | low       | T2S      |         99 |           99 |       99 |         99 |     0 |
| DOU      | high      | SWI      |         20 |            0 |       20 |          0 |    74 |
| MOMENI   | low       | SWI      |        118 |          100 |      370 |        313 |   146 |
| RODEJA   | high      | SWI      |         88 |           32 |       88 |         32 |   286 |
| RODEJA   | low       | SWI      |         15 |           10 |       15 |         10 |    71 |
| VALDO    | high      | T2S      |         45 |           22 |       45 |         22 |   219 |
| VALDO    | low       | T2S      |         27 |            0 |       27 |          0 |    34 |
| sMOMENI  | low       | SWI      |        118 |            0 |     3700 |          0 | 36812 |
| Total    | -         | -        |       1027 |          897 |     5015 |       1119 | 37738 |

* Summary table with more info for every dataset:

| Dataset      | Study        | Demographics                                                                                | Location                   | Field Strength | Scanner Model                                                                                                                 | Flip Angle | TR      | TE      | Slice Thickness | Rating Scale            | TR/TE           |
|--------------|--------------|---------------------------------------------------------------------------------------------|----------------------------|----------------|-------------------------------------------------------------------------------------------------------------------------------|------------|---------|---------|-----------------|-------------------------|-----------------|
| VALDO        | VALDO-SABRE  | Tri-ethnic, high cardiovascular risk, 36-92 years old, mean age 72.                         | London, UK                 | 3              | Philips                                                                                                                       | 18         | 1288    | 21      | 3               | BOMBS                   | 1288/21         |
| VALDO        | VALDO-RSS    | Aging population >45 without dementia                                                       | Rotterdam, Netherlands     | 1.5            | GE MRI                                                                                                                        | 13         | 45      | 31      | 0.8             | (Vernooij et al., 2008) | 45/31           |
| VALDO        | VALDO-ALFA   | Enriched for APOE4, family risk of Alzheimer’s. Cognitively normal participants aged 45-74 | Barcelona, Spain           | 3              | GE Discovery                                                                                                                  | 15         | 1300    | 23      | 3               | BOMBS                   | 1300/23         |
| RODEJA       | RODEJA       | Unknown                                                                                     | Copenhagen region, Denmark | 1.5/3          | Several                                                                                                                       | Several    | Several | Several | Several         | Unknown                 | Several/Several |
| CEREBRIU     | CEREBRIU     | Unknown                                                                                     | Brazil                     | 1.5            | GE Optima MR450w 1.5                                                                                                          | 15         | 75      | 48      | NA              | BOMBS                   | 75/48           |
| CEREBRIU     | CEREBRIU     | Unknown                                                                                     | India                      | 1.5            | Siemens Symphony 1.5                                                                                                          | 12         | 48      | 40      | NA              | BOMBS                   | 48/40           |
| CEREBRIU     | CEREBRIU     | Unknown                                                                                     | India                      | 1.5/3          | 33% Siemens MAGNETOM Sempra 1.5, 33% Siemens Sempra 1.5, 33% Siemens Spectra 3                                                | 20         | 688-872 | 19-25   | NA              | BOMBS                   | 688-872/19-25   |
| CEREBRIU     | CEREBRIU     | Unknown                                                                                     | U.S.A                      | 3              | Siemens MAGNETOM Vida 3                                                                                                       | 15         | 27      | 20      | NA              | BOMBS                   | 27/20           |
| CEREBRIU_neg | CEREBRIU_neg | Unknown                                                                                     | Brazil                     | 1.5            | 34% GE Brivo MR355 1.5, 31% GE GENESIS_SIGNA 1.5, 35% GE Optima MR450w 1.5                                                    | 15-20      | 74-717  | 19-50   | NA              | BOMBS                   | 74-717/19-50    |
| CEREBRIU_neg | CEREBRIU_neg | Unknown                                                                                     | India                      | 1.5/3          | 50% Siemens MAGNETOM_ESSENZA 1.5, 50% Siemens Spectra 3                                                                       | 20         | 657-762 | 19      | NA              | BOMBS                   | 657-762/19      |
| CEREBRIU_neg | CEREBRIU_neg | Unknown                                                                                     | India                      | 1.5            | 56% Siemens MAGNETOM_ESSENZA 1.5, 44% Siemens Symphony 1.5                                                                    | 12-20      | 48-688  | 20-40   | NA              | BOMBS                   | 48-688/20-40    |
| CEREBRIU_neg | CEREBRIU_neg | Unknown                                                                                     | India                      | 1.5/3          | 53% Siemens MAGNETOM Sempra 1.5, 3% Siemens MAGNETOM_ESSENZA 1.5, 5% Siemens NA 1.5, 26% Siemens Sempra 1.5, 14% Siemens Spectra 3 | 20         | 657-968 | 19-25   | NA              | BOMBS                   | 657-968/19-25   |
| CEREBRIU_neg | CEREBRIU_neg | Unknown                                                                                     | U.S.A                      | 1.5/3          | 7% Siemens MAGNETOM Sola 1.5, 68% Siemens MAGNETOM Vida 3, 3% Siemens NA 3, 20% Siemens Prisma_fit 3, 1% Siemens TrioTim 3    | 15-20      | 27-782  | 20-40   | NA              | BOMBS                   | 27-782/20-40    |
| MOMENI       | MOMENI       | Alzheimer’s disease (AD), mild cognitive impairment (MCI) and cognitively normal (CN)       | Australia                  | 3              | Siemens TRIM TRIO scanner                                                                                                     | 20         | 27      | 20      | 1.75            | MARS                    | 27/20           |
| DOU          | DOU          | 10 cases with stroke and 10 cases of normal aging                                           | China?                     | 3              | Philips Medical System                                                                                                        | 20         | 17      | 24      | 2               | MARS                    | 17/24           |
| sMOMENI      | MOMENI       | Alzheimer’s disease (AD), mild cognitive impairment (MCI) and cognitively normal (CN)       | Australia                  | 3              | Siemens TRIM TRIO scanner                                                                                                     | 20         | 27      | 20      | 1.75            | MARS                    | 27/20           |


* My idea is to stratify by seq type (SWI/T2S), original resolution level and field strength. Maybe leave some full dataset like DOU out (20 cases, also publicly avaialble) as test set.

<br><br><br><br><br>




### Date: 18 March 2024

#### What did you achieve/do?
* Preprocessing finished. Also renamed all studies with nice id to facilitate tracking experiments
* Adapted metadata to be ingested correctly by ClearML
* Created splits file both for pre-training and fine-tuning phase (more details after)
* Started configuring ClearML for my pretraining task

#### What did you struggle with?
* ClearML bugs in code

#### What would you like to work on next ?
* Run several exepriments to choose best set of hyperparams: patch size, loss function, ...etc
* Add data augmentations 

#### Where do you need help from Veronika?
* I will start writing report this week in Overleaf. Because i intend to submit to some MICCAI Workshop, I wanted to follow its format already. Hoever I see it is Springer format iwth a lot of margins and one-column, so maybe it is not fititng ITU guidelines? Would you abandon this idea? 

#### Others:

#### Splits file
* Splitting of studies has been somewhat tricky. I did the split straified by the following conditions: 
- field strength
- resolution level
- sequence type
- source of data (dataset)
- healthy/unhealthy
- cmb per case (<=5, >5)

* I made sure that same patient was never in both trianing and validaiton set.
* 25 - 75 % valid-train split
* For pre-training, I will add ONLY to trianing set the CRB negative samples (which I collected form internal database and coudl maybe not be fully negative) and the synthetic CMBs.  

<br><br><br><br><br>




### Date: 25 March 2024

#### What did you achieve/do?
* Started training for several experiments using Apollo architecture (the 3D Unet-based
architecture that they use for the company's software). 

* Pretraining phase experiment:
    - Pretraining of Apollo from scratch on negative-and-synthetic-enriched dataset
    - Same as the previous but starting with Apollo pretrained weights

* VALDO experiments:
    - Apollo from scratch performance on VALDO only
    - Apollo pre-trained performance on VALDO only

* No-pretraining experiments:
    - Training of Apollo from scratch on dataset WITHOUT negative-and-synthetic cases
    - Same as the previous but starting with Apollo pretrained weights

All configs used for all experiments found in here: [experiments folder](../experiments)


#### What did you struggle with?
* I had to deal with a lot of debugging for running the experiments before having something up and running
* I had to extract manually the relevant pretrained weights from Apollo (bcs it originally works with 3 input channels for different sequence types; and 5 output labels, for several pathologies), to keep only weights for T2S/SWI and for MACROhemorrhages


#### What would you like to work on next ?
* Overleaf report writing of:
    - Label Refinement
    - Data Preprocessing
    - Background

* Literature research on Transfer Learning
* Research on the possible use of YOLO architecture (and how ot implement) to compare results

#### Where do you need help from Veronika?
* For the report, should the structure be the same as Research Project, or more/different sections are expected
* Do you want to set some date for report revision like last time? Can we set some sort of pre-revision date?(to agree on structure at least, would be very helpful)
* Can you recommend me nice papers on transfer learning?

#### Others

* n/a

<br><br><br><br><br>




### Date: 01 April 2024

#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>




### Date: 08 April 2024

#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>




### Date: 15 April 2024

#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>




### Date: 22 April 2024

#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>




### Date: 29 April 2024

#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>




### Date: 06 May 2024

#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>




### Date: 13 May 2024

#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>




### Date: 20 May 2024

#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>




### Date: 27 May 2024

#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>




### Date: 03 June 2024

#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>




### Date: 10 June 2024

#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>




### Date: 17 June 2024

#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>




### Date: 24 June 2024

#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>




### Date: 01 July 2024

#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>