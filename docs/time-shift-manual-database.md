# Time Shifted Selections

The original manual database did not have any time variance in the selections. To introduce this, 
I added a shift of 0.3 with a minimum overlap of 1, so that the selection could be shifted forwards 
or backwards by 0.3 in the selection window, as long as the full signal was still present in the 
one-second duration spectrogram. 

| Site    | Original | Shifted | Diff |
| ------- | -------- | ------- | ---- |
| CB      | 185      | 408     | 223  |
| ULU2022 | 1883     | 3187    | 1304 |
| ULU     | 901      | 2237    | 1336 |
| KK      | 1749     | 4069    | 2320 |
| PP      | 71       | 198     | 127  |

Need to generate that many more noise segments per site. 

For now, just regenerated all noise segments. 

Dropped vals: 







