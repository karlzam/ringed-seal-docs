# Annotations

## General Rules:
- Bearded seal overlapping calls: keep when the bearded seal call is above 1kHz 
- Do not include yelps on their own 
- Yelps in between barks can be kept, but no use keeping yelps that are on their own 
- If a bark level is hardly distinguishable from the background noise, do not keep 
- Add an overlap column to distinguish really good calls from not so good calls 
- Growls were lumped into the “bark” category if within previously annotated region
- If I see an unannotated call that looks similar close to a previous annotation, I’ll add it 
- Manually check noise segments once generated

## Column names: 
- KZ Keep? (Y/X/M)	
- Barks (KZ): number of barks 
- Yelps (KZ): number of yelps 
- Overlap (Y/X) (KZ): this is overlap w other species based on time, note this is not always accurate for the yelps. Note my screen was set at 3000Hz so there may be signals over that not included in the overlap column, 
- Call Type (KZ): bark, yelp, or “bark, yelp”. Growls in annotated regions were labeled as barks. 
- To speed things up, changed to “B”, “Y”, or “BY” - will need to manually edit the first sites to match this notation. Added “UNK” to some that were weird for unknown. 
- Marianas annotations: Added “G” for growl/groan. Will need to update scripts for this. 
- Split From (KZ): What original annotation number it was split from. Some annotations were split before I started keeping track of this - will say “unk” for unknown in this case. “S” means this one (original annotation) was split into others and not kept. 
- KZ Comment (S means this wasn’t kept because it was split into another annotation)
- Bill Check (KZ) column: need bill to check these if Y 

## Notes for thesis about annotations:
- Sometimes yelps continue for a while, hard to distinguish the start and beginning, could be double counting
- Very much depends on the spectrogram settings used - some are not optimal and some calls would be missed 
- Human error: tired, mistakes after many annotations
- Hard to distinguish between groans/growls/barks 
- Sometimes bark groans are lumped in with bark yelps 
- Multiple seals overlapping
- Distance to seal 
- Paging sounds: maybe double counting if looking at the same sound from two files (end and beginning) 
- Water noise at equally spaced increments can look like a bark
- Water noise can also sound like a bark 
- Variability in calls themselves - different analysts would have different opinions 

## Time to complete annotations:
- July 13: Completed 6590 
- July 13-July 19: 1642 completed in 4.5 days 
- Rate at fastest: 1642/3 days = 1642/15 hours (assuming 5 good hours per day of annotations) = 109.5 annotations per hour
- Started on:  June 13th Mariana emailed me the data, 4 ish weeks worked on these, 5 hours per day at 5 days per week, sooo 25 hours/week, for a total of 100 hours. 
- Total number of annotations: 8153 annotations 
- Total rate: 8153/100 hours = 81.53 annotations/hour for 100 hours. 

## Guideline Doc I Wrote

This document aims to provide instructions for creating ringed seal annotations using Raven for use with deep
learning detectors. Currently the detector is aimed at barks.

### General Guidelines
- If longer than one second (somewhat arbitrarily chosen) in between sounds, make multiple annotations - do not
draw a large box around multiple B/BY’s if gap is larger than 1s
- Do not include yelps on their own
- Include yelps in between barks, counts as a sound for the one second gap (ie. bark-yelp-bark counts if its at
0s-1s-2s)
- Important parts are the start and end time, not necessarily the frequency (although nice to have for statistics)
- Growls and groans were labelled as ”B” and added a comment when nearby or within previous annotations,
need to decide what to do in the future for these call types
- Keep track of the spectrogram parameters as you go (general rough guidelines are great)
- If you are unsure, mark the annotation has X but keep the call type as ”B”, ”BY”, or ”Y” for what you think
it is. This way I can remove these from the negative annotations.
- Annotate all sounds in all files even if they’re not ideal - this way they won’t be added into the negative segments
FIG. 1. Annotation Boxes

### Raven Instructions
Necessary Columns:

1. Keep for detector? (Y/X/M): M for maybe if it needs to be double checked

2. Overlap (Y/X): Is there overlapping signals (boats, bearded seals, etc)? If below 1kHz, do not keep. If above
1kHz, keep, but add a comment. Helps to distinguish good calls from not so good calls.

3. Call Type: ”B”, ”BY”, or ”Y” for ”bark”, ”bark, yelp”, and ”yelp”, respectively. Groans/grunts are labelled
as B.

4. Comment: whatever you’d like to comment, and/or the type of overlap

5. Barks: (not necessary but nice for stats) number of barks within the annotation

6. Yelps: (not necessary but nice for stats) number of yelps within the annotation

### HOW ARE THEY USED?
The deep learning detector will be a ”presence vs. absence” tool, meaning that for an input spectrogram the
detector will classify the spectrogram as containing a ringed seal bark, or not. The Raven annotation tables are used
to create training, validation, and testing data sets which are composed of spectrograms.
The steps to create these data sets from the annotation tables are:

1. I take the raven tables, and using an automated script, create ketos format tables, and split those tables into
training/validation/testing csv files

2. A spectrogram duration is chosen by the programmer. Currently I am using 2 seconds, but this is subject to
change.

3. Based on the annotation tables, positive and negative segments are created. This is done by splitting up the
entire wav file into 2 second clips, with a specified overlap so that calls sitting exactly on 2 seconds are not
missed.

4. Positive segments are labelled based on the annotation table - ie. if the start and end time is within the 2 second
block, that spectrogram is labelled ”ringed seal”

5. Negative segments are generated automatically by selecting random segments that are NOT annotated. This is
an important point - any calls not annotated can potentially be mistaken for ”noise”. Currently this is the case
as there as some unannotated barks, and so each time a new database is created, the noise segments need to be
manually confirmed.

6. Once generated, the positive and negative spectrograms are saved in a ”database.h5” file

7. This database file is used to train the detector