import re
_ABBREVS = r"(?:et al|Fig|Figs|Eq|Eqs|Dr|Prof|Mr|Mrs|Ms|Jr|Sr|vs|i\.e|e\.g|cf|approx|Ref|Refs|Vol|No|Ch|Sec|pp)"
# original pattern (errors)
# pattern = rf'(?<!{_ABBREVS})(?<=[.!?])\s+'

text = "This is a sentence. See Fig. 1 for details. Also et al. said this! Is it e.g. true? Yes."

# Alternative 1: negative lookbehind with fixed width
# Instead of lookbehind, we can just find all sentence boundaries
# Actually, since lookbehind must be fixed width, we can't use it easily.
# But we can do this:
# Replace periods in abbreviations with a placeholder, split, then restore.
# Or just use regex to match sentences directly.
