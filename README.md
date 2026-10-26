# Taikyoku Shogi

This is a repo that originated from an extremely stupid idea, of making a full fledged implementation of the original [Taikyoku Shogi](https://en.wikipedia.org/wiki/Taikyoku_shogi). The game is a **36 x 36** variant of regular shogi on a large board with 301 piece types and 402 pieces per side including duplicates. The Japanese monks who had designed this games also had good foresight in disabling "drops" as in regular shogi, so that the already ridiculous runtime of 20 hours did not burgeon further into 200 hours.

My implementation differs from a faithful transcription in a few ways:
- *Crown prince is not considered a royal piece.* After several hours of frustration in trying to implement the original correctly, I decided for the sake of mental health to not count crown prince as a royal piece.
- **Crown prince does not promote.* As above, the base game has crown prince promote into another King. This once again brings back the headache of detecting game-ends.
- *There is no check.* This is an ambiguity in the rules, where check was not mandatory to respond to. I decided to go without the check, as the other option would incur a performance penalty when running the program's legal move generator.
- *There is no proper end screen.* While occupied with everything else, I did not end up getting around to properly setup an ending screen. I deemed this to be not a significant issue as typically the game length to get to the ending is somewhere around 1000~1500 plies.

Overall, I don't expect anyone to actually play this or look at it for anything other than for brief curiosity. I mostly made this project because frankly, I thought it was funny. It also proved to me that despite my tendency to leave around 20 unfinished projects per year, I was in fact capable of finishing a project.

### To run:
Open `index.html` directly in the browser. No compilation is necessary.

## What is currently in this repo
A complete implementation of the board is in `game.js`.

Every piece ID, movement, and promotion was hand-typed and also in `game.js`.

Most of the UI code is in the increasingly inaccurately named `grid.js`.

In addition, the board viewer now allows you to export and import `.tkn` files (the format used for this project's Taikyoku Shogi games) to step through each individual move. (`sample_tiny.tkn` is a small 8-ply example game file, and `sample_checkmate.tkn` is a 433-ply file pulled from the self-play pipeline that resulted in checkmate.)

## What I hope to improve in the future
I had half-implemented an `engine.js` file, but removed it in the latest commit as the architecture change required a much more sophisticated implementation of search in javascript that I do not have the timeline, nor motivation, nor knowledge of ONNX to implement. As it is a highly skill based implementation challenge, I did not feel like AI-generating the code was helpful to my learning or the project.

There is still no formal end screen. This is partly because the project was a little rushed at the end to try to fit in Stardance durations, but also I must admit the role of my serial procrastination in the lack of time.

Aside from that, the layout is still quite primitive. As I have not taken any sort of web design course, I could only try my best to pick colors which were not overly grating on the eyes. In the future, I would like to revisit this project and polish it up a bit more.


# Engine

A large amount of time was spent on attempting to train a value-and-policy engine to play this game. While this did not produce an adequately strong engine for now, the process of designing the architecture, dealing with the issues that came up, and discerning what benchmarks needed to be coded was highly educational for me.

For more information on the training steps, [`engine/README.md`](engine/README.md) has more information. (Note that the engine README file is written by Claude Code and not me.)

The engine architecture is as the following:
```
Input(36×36) 
    ↓ 
Embedding(V=1205, d=64) 
    ↓ (64×36×36) 
Conv2d(64=>96, 3×3) + GroupNorm(12) + ReLU
    ↓ (96×36×36) -------------------------------------------┐
Conv2d(96=>128, 3×3, stride=2) + GroupNorm(16) + ReLU       |
    ↓ (128×18×18)                                           |
Flatten                                                     |
    ↓ (324×128)                                             |
+ PosEmbedding(324×128)                                     |
    ↓ (324×128)                                             |
TransformerEncoder(d=128, heads=4, ff=512, pre-norm)        |
    ↓ (324×128)                                             |
TransformerEncoder(d=128, heads=4, ff=512, pre-norm)        |
    ↓ (324×128)                                             |
TransformerEncoder(d=128, heads=4, ff=512, pre-norm)        |
    ↓ (324×128) --------------------┐                       |
Mean1d                              |                       |
    ↓ (128)                         ↓                       |
Linear(128=>128) + ReLU      Linear(128=>256)               |
    ↓ (128)                         ↓ (324×256)             |
Linear(128=>1) + tanh        PixelShuffle(2)                |
    ↓                               ↓ (64×36×36)            |
Value (1)                        concat() ⟵-----------------┘
                                    ↓ (160×36×36)
                                Conv2d(160=>64, 1×1)
                                    ↓ (64×36×36)
                                Flatten
                                    ↓ 
                                sq (1296×64)

Policy:                                
for each move (from, to, type):
    q = Linear(64=>32)(sq[from]) + Embedding(V=10, d=32)(type)
    k = Linear(64=>32)(sq[to])
    logit = q · k
```

# AI use
I used AI for probably a far larger portion of programming than is ideal. Most of `engine/` files are contributed to by Claude Code as I do not know how to write C. AI also helped with some stylistic features in html/css which I did not know how to implement.

Despite this, I would say that I had done much more effective work and also learnt more from reading AI output than I could have done by stubbornly pushing on by myself. As a possible rabbit-who-suffers-from-ADHD, I do not doubt that the pain of doing everything by myself would have driven me away quite early through the development. 

Even just the menial data-entry task took nearly a month and a manual overhaul arc (I had realized at the time that the architecture I was building it on was not scalable for a board viewer at all, and had to reprogram the entire board data structure) for me to "get on with it" and actually start even the interesting part. I would attribute a large part of the reason I was able to keep the motivation to continue this project was because I knew that I had a helper.