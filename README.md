
## Mapping between `sounddevice` and `whisper` terminology

##

<!-- test this again, writing it out makes this not seem right (there's no way I accumulate 20 GB of RAM here) -->
<!-- . However, accumulating more than 5 seconds of raw audio input causes my machine to run out of RAM (32 GB on my system).   -->

## Audio stapling

:

<basic illustration of sentence divided into chunks with word being segmented>

Transcribing each audio chunk and concatenating the results produces poor transcriptions. 

<example>

Since the audio is split into multiple chunks that are transcribed separately, relying solely sliding window solutions like the one implemented in Whisper (section 4.5 of their paper) will not help us. Moreover, since we want to be able to plug in any mdoel of choice into the pipeline, we want our solution to also be model-agnostic.

To resolve this, I truncate and concatenate : 

1. 

My guess is that this has been done in audio before and I didn't find it during R&D. (I spent a lot more time on R&D and planning than I should've - overplanning is a common issue for me! - so I decided not to dig deeper.) If I find a more exact name, I'll update this section.

##

## Considered features

<!-- - Instead of hardcoding models, use HuggingFace pipelines / other . I pivoted away from this when I realized I needed to use lower-level functionality  -->
