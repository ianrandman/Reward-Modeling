This directory stores all of the user preferences for each environment.  In each directory is a singular json file named
pref_db.json. This is a file containing a singular json list of triples with the following objects in the triple: seq1,
seq2, and p. seq1 and seq2 are lists of (observation, action) pairs, and p is the user preference (1 if they chose seq1,
0 if they chose seq2, and 0.5 if they chose "can't tell").