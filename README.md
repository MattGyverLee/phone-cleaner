# phone-cleaner
 
I've started this new repository: https://github.com/MattGyverLee/phone-cleaner/ . 

I'm using python and Jupyter notebooks for testing. You just set the options you want at the top (I suggest defaults) and run the whole notebook so you can examine the results. The images and data dumps are output to folders, but you may need to create four new empty folders off the repo root after cloning the repo so it won't error.

    profiles/*
    attested_components/*
    components/*
    temp/*

Please let me know if you have questions/ideas, or if this work even seems helpful. TargetPhones.ipynb is my main workplace, where I seek to work out some intelligent ways to filter the Phoible data for our usage.

- [x] I import just over 4000 phones (and allophones) from Phoible, calculate cross-language frequency for each phone, and remove duplicates.
- [x] I have currently configurable options to remove all diacritics, filter multiphone strings, tone, filter by frequency, and test those results.
- [x] After cleaning the things I want to filter, I use panphon to filter out invalid phones (I'm still working through why about 400 of the 4000 are marked as invalid (or maybe just unknown), but I see some patterns).
- [x] After filtering, I use phonetic distance to look for phones that are transcribed differently, but equivalent in phonetic features and group them together (and choose a paradigm form).  /ɓ/, /bˀ/, /ˀb/, and /ˀɓ/ are one such example of a group with the four transcriptions and the same phonetic features.
- [x] Then I use Panphon to calculate a weighted phonetic distance (feature changes required to transition). This can be used in later calculations.
- [x] (This is the part I'm most proud of.) I group the phones by that phonetic distance and generate a dendrogram of phonetically similar segments. This could be the foundation of "correcting" phones to nearby phones.
  - [x] The dendrogram and phonetic distance are a good starting point for phonetics, but we'll probably want to re-weight it features that are more audible/sonorous. For example, voicing is only one feature change (so low weighted difference between the voiced and unvoiced version of a phone), but Allosaurus seems to be unlikely to confuse /d/ and /t/ which are very different on a spectrograph. Voicing should probably be re-weighted higher.  /k/ and /q/ are less sonorant and might be more likely to be confused, so that weighting could go down. 
  - [x] Maybe I need to generate all of these individual isolated phonemes with eSpeak (via a conversion to xSampa) to give me a paradigm audio form, convert them to MFCCs in Praat or Python and diff them to see which features are more audible for re-weighting them.
- [x] I produce a Sorted Heatmap of all of the phones showing their phonetic similarity.
- [x] Then I output the proposed data to text or CSV files.

- [ ] I'm still trying to understand the last of the diacritics and phonetic modifications to decide where to class/weight and whether we can filter them.

Later, I will apply the same results/logic on a reparse of the Phoible data to :

- [ ] List the phones and allophones in each language. (Eventually, we may need to generate a phone list for our target languages if not in Phoible and an Epitran conversion script.)
- [ ] Use IPAPy and Panphon to generate a master table with
  - [ ] Phoneme names for each (for example: "near-open front unrounded vowel") .
  - [ ] Generate an x-Sampa equivalent for each phone if needed.
  - [ ] Generate Feature Charts for the linguists from Panphon.
  - [ ] Unicode Values
  - [ ] Other things of use.
- [ ] Use IPAPy to cross-check the validity of PanPhon's assessments. So far, I've seen 98%+ agreement, and IPAPy only kicks out a few extra phones...but the interesting ones will be where they disagree.
- [ ] Correct or simplify invalid phones in language profiles as above.
- [ ] Verify that individual "building blocks" are defined. Phoible hyperspecifies, so for example, you always find clicks with a previous marker for place of articulation.
- [ ] Create a master list of "roughly equivalent" phones (including phones with filtered diacritics) so that the training data can be "simplified" on import.
- [ ] Prepare to use that list of "roughly equivalent" phones to eventually allow the user to see their preferred transcription style. 
- [ ] This will be the source data to calculate the minimum languages ( I don't know how do do that calculation efficiently). I can already see that most of the "clicks" come from one or two languages, so dropping sounds in only one language could be very counterproductive.  

The old code in ParsePhoible.ipynb (quickly written overnight in 2018) still works, but I've rewritten the cleaners in a new file and will gradually move the language profile export under the TargetPhones Notebook.

Notes.ipynb is where I'm learning to use the libraries without messing up the main notebook.
