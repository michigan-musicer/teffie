
## Introduction

This . 

The main takeaways of this project include:
- The code is entirely local - no server setup required. 
- (stats for performance given current specs)

##


<!-- ## Mapping between `sounddevice` and `whisper` terminology -->

## Why multithread?

<!-- test this again, writing it out makes this not seem right (there's no way I accumulate 20 GB of RAM here) -->
<!-- . However, accumulating more than 5 seconds of raw audio input causes my machine to run out of RAM (32 GB on my system).   -->

I went into this project assuming that . If you glance at the options for live Whisper transcription, this seems like a reasonable assumption; most projects run client/server code instead of containing everything locally. 



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

- Instead of hardcoding to specific models, use HuggingFace pipelines / other more generalized to plug in any model of choice. I pivoted away from this because it was easier to optimize for speed with specific models. However, I think it's theoretically possible to optimize using a HuggingFace pipeline.
- [Quantization](https://huggingface.co/docs/optimum/en/concept_guides/quantization) reduces the complexity of models by converting floating point weights to long or int type, which is of particular importance on weaker hardware. According to this GitHub discussion, this could offer significant speedup and/or upgrade to [larger, higher-performance models](https://github.com/openai/whisper/discussions/454)
- Creating a submodule for a whisper fork that doesn't hardcode 30 seconds for the input tensors and sample rate but instead takes them from a config file. This may help reduce computation time or this may utterly screw with Whisper.

