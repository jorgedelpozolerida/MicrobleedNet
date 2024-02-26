# Jorge's Weekly Meeting Notes

* [05 February 2024](#date-05-february-2024)
* [19 February 2024](#date-19-february-2024)
* [26 February 2024](#date-26-february-2024)

<!-- *
* [12 February 2024](#date-12-february-2024)
* [26 February 2024](#date-26-february-2024)
* [04 March 2024](#date-04-march-2024)
* [11 March 2024](#date-11-march-2024)
* [18 March 2024](#date-18-march-2024)
* [25 March 2024](#date-25-march-2024)
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




### Date: 11 March 2024

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




### Date: 18 March 2024

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




### Date: 25 March 2024

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