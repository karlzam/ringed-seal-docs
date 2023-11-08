# Manual Dataset V1

## Process 

### Logic

- The data available in this project is both spatially and temporally biased 
  - Ulu2022 data is the best quality and represents the highest number of samples for that year and site 
  - Ulu2022 also was not a long term deployment which makes it different than the rest of the sites 
  - The rest of the sites (including Ulu 2017/2018) are long term deployments with highly variable data quality 
- So, create each dataset with the most CB/KK/Ulu1718 data as possible, topping up with Ulu 2022
- This will ensure the generalizability of the detector is maximized wrt to the available data
- Pierce point is being left out of the initial training process 
  - This is because then we will have a new site to test the fully trained detector to discuss the generalizability
  - Then we can do some fine-tuning of the model and discuss that the finished product that others can use is the fully trained model with the fine-tuning process 

#### How to do the topping up? 

- I decided to find the percentage split between all data exluding ulu2022 vs. ulu 2022
- All other data represents 68%, ulu2022 represents 32%
- So, in each dataset (train, val, test) use this same split to make sure the max of the other data is available, and then topped up with ulu2022

The total numbers end up being: 

|       | Ulu 2022 | Rest | Total |
| ----- | -------- | ---- | ----- |
| Train | 943      | 1998 | 2941  |
| Val   | 268      | 570  | 838   |
| Test  | 134      | 284  | 418   |


#### Time Dependence

Can we also account for time dependence in the rest of the sites? 

  - The ulu2022 data is all may, so exclude from this thought process
  - Ulu2022 can do random split for train/val/test 

Distributions:
- CB is positively skewed 
- KK is negatively skewed 
- Ulu is pretty evenly distributed after removing 2022 data 

I think what I'll do is pick a time for each site that makes the correct # of annots. 


|         | Rest | Div 3  |
|---------|------|--------|
| Train   | 1998 | 666    |
| Val     | 570  | 190    |
| Test    | 284  | 94.667 |
| Total   | 2852 |        |

So each of the three additional sites will be split: 

|       | CB  | KK  | Ulu 2017 |
| ----- | --- | --- | -------- |
| Train | 667 | 667 | 666      |
| Val   | 190 | 190 | 190      |
| Test  | 94  | 94  | 94       |



### Steps 

1. Determine the number of annotations per month, per site 
2. Exclude PP
3. Determine the total number of annotations per site 
4. Isolate ulu2022 data as it's the best and most numerous

|                                    |      | CB  | KK   | ULU | ULU22 | CB/KK/ULU |
| ---------------------------------- | ---- | --- | ---- | --- | ----- | --------- |
| Total # of annotations (w out PP): | 4197 | 185 | 1749 | 901 | 1362  | 2835      |

![month](images\total annot split.png)

![month](images\ulu22vsrest.png)

5. Determine that ulu2022 data makes up for 32% of the annotations, and the rest make up for 68%
6. Recreate this split in the train, val, and test datasets 

| Sets  |     | \# of annots required | rnd  | ULU 2022 # | RD   | REST #  | RD   |
| ----- | --- | --------------------- | ---- | ---------- | ---- | ------- | ---- |
| Train | 0.7 | 2937.9                | 2939 | 940.48     | 943  | 1998.52 | 1998 |
| Val   | 0.2 | 839.4                 | 839  | 268.48     | 268  | 570.52  | 570  |
| Test  | 0.1 | 419.7                 | 419  | 134.08     | 134  | 284.92  | 284  |
|       |     |                       | 4197 |            | 1345 |         | 2852 |

and cleaned up a bit 

|       | Ulu 2022 | Rest | Total |
| ----- | -------- | ---- | ----- |
| Train | 943      | 1998 | 2941  |
| Val   | 268      | 570  | 838   |
| Test  | 134      | 284  | 418   |
|       |          |      | 4197  |

7. Now that we've determined the numbers of each, how do we also account for time dependence? 



## Initial Thoughts

There are two main objectives with this detector: 

1. Create a detector that works to detect ringed seal vocalizations in existing data
2. Create a detector that works on new sites (ie. a detector with good generalizability)

To start: Ulu, KK, and CB will be used and PP will be left out. 

- Ulu has the best quality data and the most data 
- KK has the second-best quality data, although much worse than Ulu 
- CB has the worst quality data 
- PP has ok quality but very low sample size 
- The data from different sites can not be assumed to be identical due to the large variation in quality 

Steps for first manual database:

1. A high percentage of KK data will be used in the training set as it is of lower quality than Ulu 
2. To top off the split, Ulu data will be used after the majority of KK data has been included
3. The rest of the Ulu data will be used for validation and testing 
4. CB will be split evenly through the sets (train, validate, test) to try to provide low quality samples 
5. PP will be used to test if the detector can detect barks in new sites, and can fine tune existing model to see if can get better results 

Fine-Tuning w PP Notes:

- Fabio found in other detectors that unfreezing the last layer, or few layers, and running a few (2-3) epochs with a new site can fine-tune the model for that site

|            | **total-annot-#** | **% **  | **b-annots** | **% **  | **by-annots** | **% **  | **total-#-barks** | **% **  | **total-#-yelps** | **% **  |
|------------|:-----------------:|---------|:------------:|---------|:-------------:|---------|:-----------------:|---------|:-----------------:|---------|
| **ulu**    | 2263              | 53      | 1326         | 47      | 937           | 64      | 3764              | 55      | 1660              | 64      |
| **kk**     | 1749              | 41      | 1318         | 47      | 431           | 29      | 2746              | 40      | 743               | 29      |
| **pp**     | 71                | 2       | 40           | 1       | 31            | 2       | 123               | 2       | 79                | 3       |
| **cb**     | 185               | 4       | 122          | 4       | 63            | 4       | 260               | 4       | 108               | 4       |
| **totals** | **4268**          | **100** | **2806**     | **100** | **1462**      | **100** | **6893**          | **100** | **2590**          | **100** |

Without pp

|        | total-annot-# | %        | b-annots | %        | by-annots | %        | total-#-barks | %        | total-#-yelps |
| ------ | ------------- | -------- | -------- | -------- | --------- | -------- | ------------- | -------- | ------------- |
| ulu    | 2263          | 53.02249 | 1326     | 47.25588 | 937       | 64.09029 | 3764          | 54.60612 | 1660          | 64.09266 |
| kk     | 1749          | 40.97938 | 1318     | 46.97078 | 431       | 29.48016 | 2746          | 39.83752 | 743           | 28.68726 |
| cb     | 185           | 4.334583 | 122      | 4.347826 | 63        | 4.309166 | 260           | 3.771943 | 108           | 4.169884 |
| totals | 4197          | 98.33646 | 2766     | 98.57448 | 1431      | 97.87962 | 6770          | 98.21558 | 2511          | 96.94981 |
