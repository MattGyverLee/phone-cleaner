This is intended to remove the music from some files. 
The script .music/quietest segment.py scans time-aligned test files to identify the music, taking the lowest amplitude of each 1/10th of a second (outputting .music\quietest_mix_windows.wav). This file was manually cleaned to remove the last 2 artifacts to create (.music\quietest_mix_cleaned.wav).

MusicRemoval\MusicMatchRemoval.py takes whole recordings from `./source` and removes the specific music aggressively, putting the file in `.output`. 
The previous "weiner" version was less agressive and possibly less destructive.
It can be directed to a folder of files. 

Todo: reprocess .Wav files and re-snip with the python notebooks 
